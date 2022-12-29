use crate::solution::Solution;

#[derive(Clone, Debug)]
pub struct SegmentTree<'a, T> {
    reference: &'a T,
    swap_id_list: Vec<(u32, u32)>,
    swap_index_list: Vec<(usize, usize)>,
}

impl<'a, T> SegmentTree<'a, T>
where
    T: Solution,
{
    pub fn new(reference: &'a T) -> Self {
        Self {
            reference,
            swap_id_list: vec![],
            swap_index_list: vec![],
        }
    }

    pub fn undo(&mut self) {
        assert!(!self.swap_id_list.is_empty());
        self.swap_id_list.pop();
        self.swap_index_list.pop();
    }

    pub fn copy_from(&mut self, tree: &SegmentTree<'a, T>) {
        self.swap_id_list.clear();
        self.swap_index_list.clear();

        for i in 0..tree.swap_id_list.len() {
            self.swap_id_list.push(tree.swap_id_list[i]);
            self.swap_index_list.push(tree.swap_index_list[i]);
        }
    }

    fn inner_swap(&self, index: usize, from: usize, to: usize) -> usize {
        if self.inner_between(index, from, to) {
            if from <= to {
                to + from - index
            } else {
                // mod を取り除けば高速化するかも？
                (from + self.len() + to - index) % self.len()
            }
        } else {
            index
        }
    }

    fn inner_between(&self, index: usize, from: usize, to: usize) -> bool {
        if from <= to {
            from <= index && index <= to
        } else {
            from <= index || index <= to
        }
    }

    pub fn to_swap_list(self) -> Vec<(u32, u32)> {
        self.swap_id_list
    }
}

impl<'a, T> Solution for SegmentTree<'a, T>
where
    T: Solution,
{
    fn prev(&self, id: u32) -> u32 {
        let index = self.index_of(id);
        let prev_index = if index == 0 {
            self.len() - 1
        } else {
            index - 1
        };
        self.id_of(prev_index)
    }

    fn next(&self, id: u32) -> u32 {
        let index = self.index_of(id);
        let next_index = if index == self.len() - 1 {
            0
        } else {
            index + 1
        };
        self.id_of(next_index)
    }

    fn between(&self, id: u32, from: u32, to: u32) -> bool {
        let id_index = self.index_of(id);
        let from_index = self.index_of(from);
        let to_index = self.index_of(to);
        self.inner_between(id_index, from_index, to_index)
    }

    fn swap(&mut self, from: u32, to: u32) {
        // 現時点での物理的な swap 位置を覚えておく
        let index_from = self.index_of(from);
        let index_to = self.index_of(to);
        self.swap_id_list.push((from, to));
        self.swap_index_list.push((index_from, index_to));
    }

    fn len(&self) -> usize {
        self.reference.len()
    }

    // swap を self.swap_id_list[0..] の順にこなしたとき、どの index にいるのかを計算する
    fn index_of(&self, id: u32) -> usize {
        let mut index = self.reference.index_of(id);
        for &(from_index, to_index) in self.swap_index_list.iter() {
            index = self.inner_swap(index, from_index, to_index)
        }
        index
    }

    fn id_of(&self, index: usize) -> u32 {
        let mut index = index;
        for &(from_index, to_index) in self.swap_index_list.iter().rev() {
            index = self.inner_swap(index, from_index, to_index)
        }
        self.reference.id_of(index)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        array_solution::ArraySolution, segment_tree::SegmentTree, solution::Solution,
        two_level_tree_solution::TwoLeveltreeSolution,
    };
    use rand::{thread_rng, Rng};

    #[test]
    fn test_segment_list_array() {
        const SIZE: usize = 101;
        let mut solution = ArraySolution::new(SIZE);
        let solution2 = ArraySolution::new(SIZE);
        let mut segment_tree = SegmentTree::new(&solution2);

        let mut rng = thread_rng();

        let mut range_list = vec![];
        for _iter in 0..SIZE {
            let from = rng.gen_range(0..SIZE as u32);
            let to = rng.gen_range(0..SIZE as u32 - 1);
            let to = if from == to { to + 1 } else { to };

            range_list.push((from, to));
        }
        let range_list = range_list;

        for &(from, to) in range_list.iter() {
            solution.swap(from, to);
            segment_tree.swap(from, to);

            for index in 0..SIZE {
                assert_eq!(solution.id_of(index), segment_tree.id_of(index));
            }

            for id in 0..SIZE as u32 {
                assert_eq!(solution.next(id), segment_tree.next(id));
                assert_eq!(solution.prev(id), segment_tree.prev(id));
            }
        }

        for &(from, to) in range_list.iter().rev() {
            solution.swap(to, from);
            segment_tree.undo();

            for index in 0..SIZE {
                assert_eq!(solution.id_of(index), segment_tree.id_of(index));
            }

            for id in 0..SIZE as u32 {
                assert_eq!(solution.next(id), segment_tree.next(id));
                assert_eq!(solution.prev(id), segment_tree.prev(id));
            }
        }
    }

    #[test]
    fn test_segment_list_tlt() {
        const SIZE: usize = 101;
        let sol_ref = ArraySolution::new(SIZE);

        let mut solution = TwoLeveltreeSolution::<30>::new(&sol_ref);
        let solution2 = TwoLeveltreeSolution::<30>::new(&sol_ref);
        let mut segment_tree = SegmentTree::new(&solution2);

        let mut rng = thread_rng();

        let mut range_list = vec![];
        for _iter in 0..SIZE {
            let from = rng.gen_range(0..SIZE as u32);
            let to = rng.gen_range(0..SIZE as u32 - 1);
            let to = if from == to { to + 1 } else { to };

            range_list.push((from, to));
        }
        let range_list = range_list;

        for &(from, to) in range_list.iter() {
            solution.swap(from, to);
            segment_tree.swap(from, to);

            for index in 0..SIZE {
                assert_eq!(solution.id_of(index), segment_tree.id_of(index));
            }

            for id in 0..SIZE as u32 {
                assert_eq!(solution.next(id), segment_tree.next(id));
                assert_eq!(solution.prev(id), segment_tree.prev(id));
            }
        }

        for &(from, to) in range_list.iter().rev() {
            solution.swap(to, from);
            segment_tree.undo();

            for index in 0..SIZE {
                assert_eq!(solution.id_of(index), segment_tree.id_of(index));
            }

            for id in 0..SIZE as u32 {
                assert_eq!(solution.next(id), segment_tree.next(id));
                assert_eq!(solution.prev(id), segment_tree.prev(id));
            }
        }
    }
}
