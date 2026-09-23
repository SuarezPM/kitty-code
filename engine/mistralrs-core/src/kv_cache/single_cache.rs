use candle_core::{Result, Tensor};

use super::NormalCache;

#[derive(Debug, Clone)]
pub struct SingleCacheSnapshot {
    pub current_seq_len: usize,
}

#[derive(Debug, Clone)]
pub struct SingleCache {
    // all_data is an option on a Tensor, this makes it possible to only create the actual tensor
    // on the first call where the batch size is easily known.
    // Also this makes it safe to clone a KvCache that has been reset (as in it will not share
    // its internal state with the cloned instance).
    pub all_data: Option<Tensor>,
    pub dim: usize,
    pub current_seq_len: usize,
    pub capacity_seq_len: usize,
    pub max_seq_len: usize,
}

impl SingleCache {
    pub fn new(dim: usize, max_seq_len: usize, capacity_seq_len: usize) -> Self {
        Self {
            all_data: None,
            dim,
            current_seq_len: 0,
            max_seq_len,
            capacity_seq_len,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn all_data(&self) -> Option<&Tensor> {
        self.all_data.as_ref()
    }

    pub fn current_data(&self) -> Result<Option<Tensor>> {
        let data = match self.all_data.as_ref() {
            None => None,
            Some(d) => Some(d.narrow(self.dim, 0, self.current_seq_len)?),
        };
        Ok(data)
    }

    pub fn snapshot(&self) -> SingleCacheSnapshot {
        SingleCacheSnapshot {
            current_seq_len: self.current_seq_len,
        }
    }

    pub fn can_append_from_snapshot(
        &self,
        snapshot: &SingleCacheSnapshot,
        append_len: usize,
    ) -> bool {
        snapshot.current_seq_len == self.current_seq_len
            && snapshot
                .current_seq_len
                .checked_add(append_len)
                .is_some_and(|len| len <= self.max_seq_len)
    }

    pub fn rollback_to(&mut self, keep_len: usize) -> candle_core::Result<()> {
        self.set_len(keep_len)
    }

    pub fn reset(&mut self) {
        self.current_seq_len = 0;
        self.all_data = None;
    }

    pub fn try_set_len(&self, len: usize) -> candle_core::Result<()> {
        if len > self.capacity_seq_len {
            candle_core::bail!(
                "kv-cache: requested length ({}) exceeds current capacity ({})",
                len,
                self.capacity_seq_len
            );
        }
        Ok(())
    }

    pub fn set_len(&mut self, len: usize) -> candle_core::Result<()> {
        self.try_set_len(len)?;
        self.current_seq_len = len;
        Ok(())
    }

    pub fn append(&mut self, src: &Tensor) -> Result<()> {
        let seq_len = src.dim(self.dim)?;
        // This doesn't seem very idiomatic but because the creation can fail, it's tricky to use
        // self.all_data.get_or_insert_with.
        if self.all_data.is_none() {
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.capacity_seq_len;
            let ad = Tensor::zeros(shape, src.dtype(), src.device())?;
            self.all_data = Some(ad);
        };

        // One dtype per cache: later appends follow the first allocation
        // (e.g. F32 compute feeding an F16 CPU cache, or vice versa).
        // No-op when dtypes already match, so healthy paths are untouched.
        let converted;
        let src = match &self.all_data {
            Some(ad) if ad.dtype() != src.dtype() => {
                tracing::warn!(
                    from = ?src.dtype(),
                    to = ?ad.dtype(),
                    "KV cache append converts dtype to the first writer's dtype; \
                     mixed precision usually means compute and cache disagree — \
                     check ModelDType vs device mapping if quality drops"
                );
                converted = src.to_dtype(ad.dtype())?;
                &converted
            }
            _ => src,
        };

        // Expand kv cache
        if self.current_seq_len + seq_len > self.capacity_seq_len {
            let diff = self.current_seq_len + seq_len - self.capacity_seq_len;
            let n_blocks_needed = diff.div_ceil(NormalCache::CACHE_GROW_SIZE);
            self.capacity_seq_len += n_blocks_needed * NormalCache::CACHE_GROW_SIZE;
            if self.capacity_seq_len > self.max_seq_len {
                candle_core::bail!(
                    "kv-cache: requested capacity ({}) above max seq len ({})",
                    self.capacity_seq_len,
                    self.max_seq_len
                )
            }
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.capacity_seq_len;
            let ad = Tensor::zeros(shape, src.dtype(), src.device())?;
            ad.slice_set(self.all_data.as_ref().unwrap(), self.dim, 0)?;
            self.all_data = Some(ad);
        }

        let ad = self.all_data.as_mut().unwrap();

        ad.slice_set(src, self.dim, self.current_seq_len)?;
        self.current_seq_len += seq_len;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn seq(dtype: DType, start: f32) -> Tensor {
        let values: Vec<f32> = (0..8).map(|i| start + i as f32 * 0.5).collect();
        Tensor::from_vec(values, (2, 4), &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }

    fn first_five(cache: &SingleCache, rows: (usize, usize)) -> Vec<f32> {
        cache
            .current_data()
            .unwrap()
            .unwrap()
            .narrow(0, rows.0, rows.1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .take(5)
            .collect()
    }

    #[test]
    fn mixed_f32_then_f16_pins_f32_with_sane_values() {
        let mut cache = SingleCache::new(0, 64, 16);
        cache.append(&seq(DType::F32, 1.0)).unwrap();
        cache.append(&seq(DType::F16, 9.0)).unwrap();
        assert_eq!(cache.all_data().unwrap().dtype(), DType::F32);
        assert_eq!(cache.current_seq_len(), 4);
        // Top-5 sanity (not parity): converted rows match the F32 baseline
        // within f16 rounding — the cache holds values, not garbage.
        for (got, want) in first_five(&cache, (2, 2)).iter().zip([9.0f32, 9.5, 10.0, 10.5, 11.0])
        {
            assert!((got - want).abs() < 0.01, "got {got}, want {want}");
        }
    }

    #[test]
    fn mixed_f16_then_f32_pins_f16() {
        let mut cache = SingleCache::new(0, 64, 16);
        cache.append(&seq(DType::F16, 1.0)).unwrap();
        cache.append(&seq(DType::F32, 9.0)).unwrap();
        assert_eq!(cache.all_data().unwrap().dtype(), DType::F16);
        assert_eq!(cache.current_seq_len(), 4);
        for (got, want) in first_five(&cache, (2, 2)).iter().zip([9.0f32, 9.5, 10.0, 10.5, 11.0])
        {
            assert!((got - want).abs() < 0.05, "got {got}, want {want}");
        }
    }
}
