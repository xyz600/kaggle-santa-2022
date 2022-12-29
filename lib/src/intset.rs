use rand::{rngs::ThreadRng, Rng};

pub struct IntSet {
    array: Vec<u32>,
    index_of: Vec<u32>,
    index: usize,
}

impl IntSet {
    const NONE: u32 = std::u32::MAX;

    pub fn new(n: usize) -> IntSet {
        IntSet {
            array: vec![0; n],
            index_of: vec![Self::NONE; n],
            index: 0,
        }
    }

    pub fn set_all(&mut self) {
        for i in 0..self.capacity() {
            self.push(i as u32);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.index == 0
    }

    pub fn contains(&self, id: u32) -> bool {
        self.index_of[id as usize] != Self::NONE
    }

    pub fn push(&mut self, id: u32) {
        if self.contains(id) {
            return;
        }
        self.array[self.index] = id;
        self.index_of[id as usize] = self.index as u32;
        self.index += 1;
    }

    pub fn pop(&mut self) -> Option<u32> {
        if self.is_empty() {
            None
        } else {
            let last_id = self.array[self.index - 1];
            self.remove(last_id);
            Some(last_id)
        }
    }

    pub fn remove(&mut self, id: u32) {
        assert!(!self.is_empty());
        assert!(self.contains(id));

        let id_index = self.index_of[id as usize];
        let last_id = self.array[self.index - 1];
        self.array[id_index as usize] = last_id;

        self.index_of[last_id as usize] = id_index;
        self.index_of[id as usize] = Self::NONE;
        self.index -= 1;
    }

    pub fn random_select(&mut self, rng: &mut ThreadRng) -> u32 {
        assert!(!self.is_empty());
        let index = rng.gen_range(0..self.index);
        self.array[index]
    }

    pub fn len(&self) -> usize {
        self.index
    }

    pub fn capacity(&self) -> usize {
        self.array.len()
    }
}
