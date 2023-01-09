use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
};

use crate::solution::Solution;

#[derive(Clone, Debug)]
pub struct ArraySolution {
    content: Vec<u32>,
    index_of: Vec<u32>,
}

impl ArraySolution {
    pub fn new(dimension: usize) -> ArraySolution {
        ArraySolution {
            content: (0..dimension as u32).collect(),
            index_of: (0..dimension as u32).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.content.len()
    }

    pub fn from_array(content: Vec<u32>) -> ArraySolution {
        let mut index_of: Vec<u32> = vec![std::u32::MAX; content.len()];
        for (idx, id) in content.iter().enumerate() {
            index_of[*id as usize] = idx as u32;
        }
        // 全ての id が登録されている
        for idx in 0..index_of.len() {
            if index_of[idx] == std::u32::MAX {
                eprintln!("fail: {}", idx);
            }
            assert_ne!(index_of[idx], std::u32::MAX);
        }

        ArraySolution { content, index_of }
    }

    pub fn copy_from(&mut self, other: &ArraySolution) {
        self.content.copy_from_slice(&other.content);
        self.index_of.copy_from_slice(&other.index_of);
    }

    pub fn save(&self, filepath: &PathBuf) {
        let f = File::create(filepath).unwrap();
        let mut writer = BufWriter::new(f);

        let line = self
            .content
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ")
            + "\n";
        writer.write(line.as_bytes()).unwrap();
    }

    pub fn load(filepath: &PathBuf) -> ArraySolution {
        let f = File::open(filepath).unwrap();
        let mut reader = BufReader::new(f);

        let mut buffer = String::new();
        reader.read_line(&mut buffer).unwrap();
        let tokens = buffer
            .trim()
            .split(" ")
            .map(|t| t.parse::<u32>().unwrap())
            .collect();

        ArraySolution::from_array(tokens)
    }

    pub fn validate(&self) {
        // content と index の一対一対応が存在するか
        for i in 0..self.len() {
            let id = self.content[i];
            assert_eq!(self.index_of[id as usize], i as u32);
        }

        // ぐるっと1周した時、必ず一筆書きになっているか
        let mut id = 0;
        for _iter in 0..self.len() - 1 {
            let next = self.next(id);
            assert_ne!(next, id);
            id = next;
        }
        assert_eq!(self.next(id), 0);
    }
}

impl Solution for ArraySolution {
    fn prev(&self, id: u32) -> u32 {
        let index = self.index_of[id as usize];
        if index == 0 {
            self.content[self.len() - 1]
        } else {
            self.content[index as usize - 1]
        }
    }

    fn next(&self, id: u32) -> u32 {
        let index = self.index_of[id as usize] as usize;
        if index == self.len() - 1 {
            self.content[0]
        } else {
            self.content[index + 1]
        }
    }

    fn between(&self, id: u32, from: u32, to: u32) -> bool {
        let id_index = self.index_of[id as usize];
        let from_index = self.index_of[from as usize];
        let to_index = self.index_of[to as usize];
        if from_index <= to_index {
            from_index <= id_index && id_index <= to_index
        } else {
            // (id -> to -> from) || (to -> from -> id) の順になっていれば ok
            (id_index <= to_index && to_index <= from_index)
                || (to_index <= id_index && from_index <= id_index)
        }
    }

    fn swap(&mut self, from: u32, to: u32) {
        let mut from_index = self.index_of[from as usize] as usize;
        let mut to_index = self.index_of[to as usize] as usize;

        let range_size = if from_index <= to_index {
            to_index + 1 - from_index
        } else {
            to_index + 1 + self.len() - from_index
        };

        for _iter in 0..(range_size / 2) {
            let from = self.content[from_index] as usize;
            let to = self.content[to_index] as usize;
            self.index_of.swap(from, to);
            self.content.swap(from_index, to_index);
            from_index = if from_index == self.len() - 1 {
                0
            } else {
                from_index + 1
            };
            to_index = if to_index == 0 {
                self.len() - 1
            } else {
                to_index - 1
            };
        }
    }

    fn len(&self) -> usize {
        self.content.len()
    }

    fn index_of(&self, id: u32) -> usize {
        self.index_of[id as usize] as usize
    }

    fn id_of(&self, index: usize) -> u32 {
        self.content[index]
    }
}

#[cfg(test)]
mod tests {
    use crate::solution::Solution;

    use super::ArraySolution;

    #[test]
    fn test_solution_swap() {
        let dimension = 100;
        let mut solution = ArraySolution::new(dimension);

        // [0, ..., 19, 20, 21, ..., 79, 80, 81, ...]
        assert_eq!(solution.prev(19), 18);
        assert_eq!(solution.next(19), 20);
        assert_eq!(solution.prev(80), 79);
        assert_eq!(solution.next(80), 81);
        assert_eq!(solution.prev(20), 19);
        assert_eq!(solution.next(20), 21);
        assert_eq!(solution.prev(81), 80);
        assert_eq!(solution.next(81), 82);

        solution.swap(20, 80);
        // [0, ..., 19, 80, 79, ..., 21, 20, 81, ...]
        assert_eq!(solution.prev(19), 18);
        assert_eq!(solution.next(19), 80);
        assert_eq!(solution.prev(80), 19);
        assert_eq!(solution.next(80), 79);
        assert_eq!(solution.prev(20), 21);
        assert_eq!(solution.next(20), 81);
        assert_eq!(solution.prev(81), 20);
        assert_eq!(solution.next(81), 82);
    }

    #[test]
    fn test_solution_swap2() {
        let dimension = 100;
        let mut solution = ArraySolution::new(dimension);

        solution.swap(80, 20);
        // [0, 99, 98, ..., 81, 80, 21, 22, ..., 79, 20, 19, ..., 1]
        assert_eq!(solution.prev(19), 20);
        assert_eq!(solution.next(19), 18);
        assert_eq!(solution.prev(80), 81);
        assert_eq!(solution.next(80), 21);
        assert_eq!(solution.prev(20), 79);
        assert_eq!(solution.next(20), 19);
        assert_eq!(solution.prev(81), 82);
        assert_eq!(solution.next(81), 80);
    }
}
