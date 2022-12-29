pub struct BitSet {
    valid_index: u32,
    array: Vec<u32>,
}

impl BitSet {
    pub fn new(n: usize) -> BitSet {
        BitSet {
            valid_index: 0,
            array: vec![std::u32::MAX; n],
        }
    }

    pub fn len(&self) -> usize {
        self.array.len()
    }

    pub fn clear_all(&mut self) {
        self.valid_index += 1;
    }

    pub fn clear(&mut self, index: u32) {
        self.array[index as usize] += 1;
    }

    pub fn set(&mut self, index: u32) {
        self.array[index as usize] = self.valid_index;
    }

    pub fn test(&self, index: u32) -> bool {
        self.array[index as usize] == self.valid_index
    }
}

#[cfg(test)]
mod tests {
    use crate::bitset::BitSet;

    #[test]
    fn test_bitset() {
        const LEN: u32 = 128;
        let mut bs = BitSet::new(LEN as usize);
        bs.clear_all();
        for index in 0..LEN {
            assert!(!bs.test(index))
        }
        bs.set(8);
        for index in 0..LEN {
            if index == 8 {
                assert!(bs.test(index))
            } else {
                assert!(!bs.test(index))
            }
        }
    }
}
