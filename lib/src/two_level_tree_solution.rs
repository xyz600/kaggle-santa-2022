use std::cmp::Ordering;
use std::ops::{Index, IndexMut};

use crate::array_solution::ArraySolution;
use crate::solution::Solution;

#[derive(Clone, Debug)]
struct Segment<const N: usize> {
    array: [u32; N],
    // 次に挿入する場所
    index: usize,
    reversed: bool,
    // 1次元化した時に、先頭から数えて物理的に何番目に存在するかをメモする
    // swap の度に更新して、 Solution::index_of(id) へ高速にこたえるために存在する
    start_1d_index: usize,
}

impl<const N: usize> Segment<N> {
    pub fn new() -> Segment<N> {
        Segment {
            array: [0u32; N],
            index: 0,
            reversed: false,
            start_1d_index: 0,
        }
    }

    pub fn front(&self) -> u32 {
        if self.reversed {
            self.array[self.len() - 1]
        } else {
            self.array[0]
        }
    }

    pub fn back(&self) -> u32 {
        if self.reversed {
            self.array[0]
        } else {
            self.array[self.len() - 1]
        }
    }

    pub fn clear(&mut self) {
        self.reversed = false;
        self.index = 0;
    }

    pub fn push(&mut self, value: u32) {
        self.array[self.index] = value;
        self.index += 1;
    }

    pub fn pop(&mut self) {
        assert!(!self.is_empty());
        self.index -= 1;
    }

    pub fn is_empty(&self) -> bool {
        self.index == 0
    }

    pub fn len(&self) -> usize {
        self.index
    }

    pub fn swap(&mut self, idx1: usize, idx2: usize) {
        self.array.swap(idx1, idx2);
    }
}

impl<const N: usize> Index<usize> for Segment<N> {
    type Output = u32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.array[index]
    }
}

impl<const N: usize> IndexMut<usize> for Segment<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.array[index]
    }
}

#[derive(Clone, Debug)]
struct SegmentIDList {
    content: Vec<u16>,
    index_of: Vec<u16>,
    free_list: Vec<u16>,
}

impl SegmentIDList {
    const NONE: u16 = std::u16::MAX;
    fn new(n: u16) -> SegmentIDList {
        SegmentIDList {
            content: vec![],
            index_of: vec![SegmentIDList::NONE; n as usize],
            // stack として pop した時に、小さい数字が使われる傾向にあれば見やすい
            free_list: (0..n).rev().collect(),
        }
    }

    fn acquire_free_segment_id(&mut self) -> u16 {
        assert!(!self.free_list.is_empty());
        self.free_list.pop().unwrap()
    }

    fn contains(&self, segment_id: u16) -> bool {
        self.index_of[segment_id as usize] != SegmentIDList::NONE
    }

    fn push(&mut self, segment_id: u16) {
        assert!(!self.contains(segment_id));
        self.index_of[segment_id as usize] = self.content.len() as u16;
        self.content.push(segment_id);
    }

    fn remove(&mut self, segment_id: u16) {
        assert!(self.contains(segment_id));
        let remove_pos = self.index_of[segment_id as usize] as usize;
        self.content.remove(remove_pos);
        self.index_of[segment_id as usize] = SegmentIDList::NONE;
        for index in remove_pos..self.content.len() {
            let id = self.content[index];
            assert!(self.index_of[id as usize] >= 1);
            self.index_of[id as usize] -= 1;
        }
        self.free_list.push(segment_id);
    }

    // target_segment_id の前に segment_id を挿入する
    fn insert_prev(&mut self, segment_id: u16, target_segment_id: u16) {
        assert!(self.contains(target_segment_id));
        assert!(!self.contains(segment_id));
        let insert_pos = self.index_of[target_segment_id as usize] as usize;
        self.content.insert(insert_pos, segment_id);
        for index in insert_pos..self.content.len() {
            let id = self.content[index];
            self.index_of[id as usize] = index as u16;
        }
    }

    // target_segment_id の次に segment_id を挿入する
    fn insert_next(&mut self, segment_id: u16, target_segment_id: u16) {
        assert!(self.contains(target_segment_id));
        assert!(!self.contains(segment_id));
        if self.index_of[target_segment_id as usize] == self.len() as u16 - 1 {
            self.push(segment_id);
        } else {
            // 末尾以外は、insert_next(p, t) = insert_prev(p, next(t))
            let prev_target_index = self.index_of[target_segment_id as usize] as usize + 1;
            let prev_target_id = self.content[prev_target_index];
            self.insert_prev(segment_id, prev_target_id);
        }
    }

    fn next(&self, id: u16) -> u16 {
        assert!(self.contains(id));
        let index = self.index_of[id as usize] as usize;
        if index == self.len() - 1 {
            self.content[0]
        } else {
            self.content[index + 1]
        }
    }

    fn prev(&self, id: u16) -> u16 {
        assert!(self.contains(id));
        let index = self.index_of[id as usize] as usize;
        if index == 0 {
            self.content[self.len() - 1]
        } else {
            self.content[index - 1]
        }
    }

    fn swap(&mut self, from: u16, to: u16) {
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

    fn index_of(&self, id: u16) -> u16 {
        self.index_of[id as usize]
    }

    fn len(&self) -> usize {
        self.content.len()
    }

    fn validate(&self) {
        // content + free_list = capacity
        assert_eq!(
            self.content.len() + self.free_list.len(),
            self.index_of.len()
        );
        for id in self.free_list.iter() {
            assert!(self.index_of[*id as usize] == Self::NONE);
        }
        for id in self.content.iter() {
            let idx = self.index_of[*id as usize];
            assert!(idx != Self::NONE);
            assert!(self.content[idx as usize] == *id);
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
struct TwoLevelIndex {
    segment_id: u16,
    inner_id: u16,
}

#[derive(Clone, Debug)]
pub struct TwoLeveltreeSolution<const N: usize> {
    // segment の中身
    buffer: Vec<Segment<N>>,
    // 各都市の情報
    index_of: Vec<TwoLevelIndex>,
    // 格納されている segment の順番
    segment_list: SegmentIDList,
}

impl<const N: usize> TwoLeveltreeSolution<N> {
    pub fn new<T: Solution>(sol: &T) -> TwoLeveltreeSolution<N> {
        let len = sol.len();
        let mut index_of = vec![TwoLevelIndex::default(); len];

        // \sqrt{N} 個くらいの segment に分割して登録
        let no_segment = (len as f64).sqrt().ceil() as usize;
        let segment_capacity = std::u16::MAX.min(10 * no_segment as u16);
        let mut segment_list = SegmentIDList::new(segment_capacity);
        let mut buffer = vec![Segment::<N>::new(); segment_capacity as usize];

        let mut node = 0;
        let mut accumulated_segment_size = 0;
        for iter in 0..no_segment {
            let segment_id = segment_list.acquire_free_segment_id();
            let segment_size = len * (iter + 1) / no_segment - len * iter / no_segment;
            assert!(segment_size < N);
            for inner_id in 0..segment_size as u16 {
                buffer[segment_id as usize].push(node);
                index_of[node as usize].inner_id = inner_id;
                index_of[node as usize].segment_id = segment_id;
                node = sol.next(node as u32);
            }
            buffer[segment_id as usize].start_1d_index = accumulated_segment_size;
            accumulated_segment_size += segment_size;
            segment_list.push(segment_id as u16);
        }

        TwoLeveltreeSolution {
            buffer,
            index_of,
            segment_list,
        }
    }

    // [prev, id, next, ...] -> [..., prev] + [id, next, ...] として、分割された segment id の pair
    fn split(&mut self, id: u32) -> (u16, u16) {
        let index = self.index_of[id as usize];
        let segment_id = index.segment_id;
        let new_segment_id = self.segment_list.acquire_free_segment_id();

        // reverse == true の時、仮想的に末尾に id を含めるためには、物理的に先頭に id を含める必要がある
        let offset = if self.buffer[segment_id as usize].reversed {
            1
        } else {
            0
        };
        // 物理的に front / back にあるという意味であることに注意
        let front_segment_size = index.inner_id + offset;
        let all_segment_size = self.buffer[segment_id as usize].len() as u16;
        let back_segment_size = all_segment_size - front_segment_size;

        // buffer の先頭を書き換えないように、どうにか頑張る
        self.buffer[new_segment_id as usize].reversed = self.buffer[segment_id as usize].reversed;
        for idx in front_segment_size..all_segment_size {
            let id = self.buffer[segment_id as usize][idx as usize];
            self.buffer[new_segment_id as usize][(idx - front_segment_size) as usize] = id;
            self.index_of[id as usize].segment_id = new_segment_id;
            self.index_of[id as usize].inner_id = idx - front_segment_size;
        }
        self.buffer[segment_id as usize].index = front_segment_size as usize;
        self.buffer[new_segment_id as usize].index = back_segment_size as usize;

        if self.buffer[segment_id as usize].reversed {
            // buffer の後半 = sequence の前半なので、(new, cur) の segment 順序になる
            self.segment_list.insert_prev(new_segment_id, segment_id);
            (new_segment_id, segment_id)
        } else {
            // buffer の前半 = sequence の前半なので、(cur, new) の segment 順序になる
            self.segment_list.insert_next(new_segment_id, segment_id);
            (segment_id, new_segment_id)
        }
    }

    fn dissolve_reverse(&mut self, segment_id: u16) {
        let segment_id = segment_id as usize;
        assert!(self.buffer[segment_id].reversed);
        self.buffer[segment_id].reversed = false;

        let len = self.buffer[segment_id].len();
        let max_iter = self.buffer[segment_id].len() / 2;
        for i in 0..max_iter {
            self.buffer[segment_id].swap(i, len - 1 - i);
            let id1 = self.buffer[segment_id][i] as usize;
            let id2 = self.buffer[segment_id][len - 1 - i] as usize;
            self.index_of.swap(id1, id2);
        }
    }

    // (segment_id, next_segment_id) を merge して、残った segment_id を返す
    fn merge_right(&mut self, segment_id: u16) -> u16 {
        // [segment_id, next_segment_id] を merge する
        let next_segment_id = self.segment_list.next(segment_id);

        // merge 元と merge 先の reverse の状態が合っていないと辛いので、不ぞろいなら揃える
        if self.buffer[segment_id as usize].reversed
            != self.buffer[next_segment_id as usize].reversed
        {
            if self.buffer[segment_id as usize].reversed {
                self.dissolve_reverse(segment_id);
            } else {
                self.dissolve_reverse(next_segment_id);
            }
        }

        let segment_id = segment_id as usize;
        let next_segment_id = next_segment_id as usize;

        if self.buffer[segment_id].reversed {
            // next に current を詰める
            let offset = self.buffer[next_segment_id].len();
            for i in 0..self.buffer[segment_id].len() {
                let id = self.buffer[segment_id][i];
                self.buffer[next_segment_id][i + offset] = id;
                self.index_of[id as usize].segment_id = next_segment_id as u16;
                self.index_of[id as usize].inner_id = (i + offset) as u16;
            }
            self.buffer[next_segment_id].index += self.buffer[segment_id].len();
            self.segment_list.remove(segment_id as u16);
            next_segment_id as u16
        } else {
            // current に next を詰める
            let offset = self.buffer[segment_id].len();
            for i in 0..self.buffer[next_segment_id].len() {
                let id = self.buffer[next_segment_id][i];
                self.buffer[segment_id][i + offset] = id;
                self.index_of[id as usize].segment_id = segment_id as u16;
                self.index_of[id as usize].inner_id = (i + offset) as u16;
            }
            self.buffer[segment_id].index += self.buffer[next_segment_id].len();
            self.segment_list.remove(next_segment_id as u16);
            segment_id as u16
        }
    }

    // 単一セグメントのみの swap
    fn swap_in_segment(&mut self, from: u32, to: u32) {
        let from_index = self.index_of[from as usize];
        let to_index = self.index_of[to as usize];
        assert_eq!(from_index.segment_id, to_index.segment_id);
        let segment_id = from_index.segment_id;

        let (mut from_idx, mut to_idx) = if self.buffer[segment_id as usize].reversed {
            (to_index.inner_id, from_index.inner_id)
        } else {
            (from_index.inner_id, to_index.inner_id)
        };
        // Segment 内で環状に接続されているわけではないため
        assert!(from_idx <= to_idx);

        let len = to_idx + 1 - from_idx;
        for _iter in 0..len / 2 {
            self.buffer[segment_id as usize].swap(from_idx as usize, to_idx as usize);
            let from = self.buffer[segment_id as usize][from_idx as usize];
            let to = self.buffer[segment_id as usize][to_idx as usize];
            self.index_of.swap(from as usize, to as usize);
            from_idx += 1;
            to_idx -= 1;
        }
    }

    // セグメント単位でのみswap
    fn swap_aligned(&mut self, from_segment_id: u16, to_segment_id: u16) {
        let mut segment_id = from_segment_id;
        while segment_id != to_segment_id {
            self.buffer[segment_id as usize].reversed ^= true;
            segment_id = self.segment_list.next(segment_id);
        }
        self.buffer[to_segment_id as usize].reversed ^= true;

        self.segment_list.swap(from_segment_id, to_segment_id);
    }

    fn inner_index(&self, id: u32) -> (u16, u16) {
        let index = self.index_of[id as usize];
        let segment_index = self.segment_list.index_of(index.segment_id);
        let exact_inner_index = if self.buffer[index.segment_id as usize].reversed {
            self.buffer[index.segment_id as usize].len() as u16 - 1 - index.inner_id
        } else {
            index.inner_id
        };
        (segment_index, exact_inner_index)
    }

    fn validate(&self) {
        self.segment_list.validate();
    }

    pub fn to_array_solution(&self) -> ArraySolution {
        let mut content = vec![];
        let mut id = 0;
        for _iter in 0..self.len() {
            content.push(id);
            id = self.next(id);
        }
        ArraySolution::from_array(content)
    }
}

impl<const N: usize> Solution for TwoLeveltreeSolution<N> {
    fn prev(&self, id: u32) -> u32 {
        let index = self.index_of[id as usize];
        if self.buffer[index.segment_id as usize].reversed {
            if index.inner_id == self.buffer[index.segment_id as usize].len() as u16 - 1 {
                let prev_segment_id = self.segment_list.prev(index.segment_id);
                self.buffer[prev_segment_id as usize].back()
            } else {
                self.buffer[index.segment_id as usize][index.inner_id as usize + 1]
            }
        } else {
            if index.inner_id == 0 {
                let prev_segment_id = self.segment_list.prev(index.segment_id);
                self.buffer[prev_segment_id as usize].back()
            } else {
                self.buffer[index.segment_id as usize][index.inner_id as usize - 1]
            }
        }
    }

    fn next(&self, id: u32) -> u32 {
        let index = self.index_of[id as usize];
        if self.buffer[index.segment_id as usize].reversed {
            if index.inner_id == 0 {
                let next_segment_id = self.segment_list.next(index.segment_id);
                self.buffer[next_segment_id as usize].front()
            } else {
                self.buffer[index.segment_id as usize][index.inner_id as usize - 1]
            }
        } else {
            if index.inner_id == self.buffer[index.segment_id as usize].len() as u16 - 1 {
                let next_segment_id = self.segment_list.next(index.segment_id);
                self.buffer[next_segment_id as usize].front()
            } else {
                self.buffer[index.segment_id as usize][index.inner_id as usize + 1]
            }
        }
    }

    fn between(&self, id: u32, from: u32, to: u32) -> bool {
        // from, id, to の buffer の位置 && 内部位置を求める
        let from_index = self.inner_index(from);
        let id_index = self.inner_index(id);
        let to_index = self.inner_index(to);

        match from_index.cmp(&to_index) {
            Ordering::Less => {
                // from <= id <= to
                from_index.cmp(&id_index) != Ordering::Greater
                    && id_index.cmp(&to_index) != Ordering::Greater
            }
            Ordering::Equal => id == from,
            Ordering::Greater => {
                // (to <= from <= id) or
                // (id <= to <= from)
                from_index.cmp(&id_index) != Ordering::Greater
                    || id_index.cmp(&to_index) != Ordering::Greater
            }
        }
    }

    fn swap(&mut self, from: u32, to: u32) {
        let from_segment = self.index_of[from as usize].segment_id;
        let to_segment = self.index_of[to as usize].segment_id;

        if from_segment == to_segment
            && ((self.index_of[from as usize].inner_id < self.index_of[to as usize].inner_id)
                ^ self.buffer[from_segment as usize].reversed)
        {
            self.swap_in_segment(from, to);
        } else {
            // from_segment
            let _ = if self.buffer[from_segment as usize].front() == from {
                from_segment
            } else {
                let (_, new_from_segment) = self.split(from);
                new_from_segment
            };

            // from の segment 作り変えによって生じうる to_segment の変化反映
            let to_segment = self.index_of[to as usize].segment_id;

            // 物理的に末尾にない場合は流用できないので、split
            let to_segment = if self.buffer[to_segment as usize].back() == to {
                to_segment
            } else {
                // [..., prev, to, next, ...] -> [..., prev, to], [next, ...]
                let (new_to_segment, _) = self.split(self.next(to));
                new_to_segment
            };

            // from の segment 作り変えによって生じうる to_segment の変化反映
            let from_segment = self.index_of[from as usize].segment_id;

            self.swap_aligned(from_segment, to_segment);

            // [..., prev_from, from_segment, ..., to_segment, next_to, ...]
            // -> [..., prev_from, to_segment, ..., from_segment, next_to, ...]
            // from, to の segment size が小さすぎたら、それぞれ merge
            let merge_threashold = (self.len() as f64).sqrt().ceil() as usize;
            {
                let segment_id = from_segment;
                if self.buffer[segment_id as usize].len() < merge_threashold {
                    let prev = self.segment_list.prev(segment_id);
                    let prev_len = self.buffer[prev as usize].len();
                    let next = self.segment_list.next(segment_id);
                    let next_len = self.buffer[next as usize].len();
                    if prev_len < next_len {
                        self.merge_right(prev);
                    } else {
                        self.merge_right(segment_id);
                    }
                }
            }

            // from の segment 作り変えによって生じうる to_segment の変化反映
            // from を merge することで to_segment が消滅する可能性を配慮して、再計算
            let to_segment = self.index_of[to as usize].segment_id;
            {
                let segment_id = to_segment;
                if self.buffer[segment_id as usize].len() < merge_threashold {
                    let prev = self.segment_list.prev(segment_id);
                    let prev_len = self.buffer[prev as usize].len();
                    let next = self.segment_list.next(segment_id);
                    let next_len = self.buffer[next as usize].len();
                    if prev_len < next_len {
                        self.merge_right(prev);
                    } else {
                        self.merge_right(segment_id);
                    }
                }
            }
        }
        // start_1d_index の再構築
        let mut accumulated_segment_size = 0;
        for segment_id in self.segment_list.content.iter() {
            let segment_id = *segment_id as usize;
            self.buffer[segment_id].start_1d_index = accumulated_segment_size;
            accumulated_segment_size += self.buffer[segment_id].len();
        }
    }

    fn len(&self) -> usize {
        self.index_of.len()
    }

    fn index_of(&self, id: u32) -> usize {
        let id = id as usize;
        let index = self.index_of[id];
        let start_index = self.buffer[index.segment_id as usize].start_1d_index;
        let segment_size = self.buffer[index.segment_id as usize].len();
        if self.buffer[index.segment_id as usize].reversed {
            start_index + segment_size - 1 - index.inner_id as usize
        } else {
            start_index + index.inner_id as usize
        }
    }

    fn id_of(&self, index: usize) -> u32 {
        // どの segment に属しているかを二分探索
        // start_index <= index を満たす buffer が知りたい
        let mut left = 0 as usize;
        let mut right = self.segment_list.len();
        while right - left >= 2 {
            let middle = (right + left) / 2;
            let segment_id = self.segment_list.content[middle];
            if self.buffer[segment_id as usize].start_1d_index <= index {
                left = middle;
            } else {
                right = middle;
            }
        }
        let segment_id = self.segment_list.content[left];
        let inner_index = index - self.buffer[segment_id as usize].start_1d_index;
        assert!(inner_index < self.buffer[segment_id as usize].len());
        self.buffer[segment_id as usize].array[inner_index]
    }
}

#[cfg(test)]
mod tests {
    use crate::{array_solution::ArraySolution, solution::Solution};

    use super::{SegmentIDList, TwoLeveltreeSolution};

    use rand::Rng;

    #[test]
    fn test_segment_list() {
        let segment_size = 64;
        let segment_capacity = segment_size * 2;
        let mut segment_list = SegmentIDList::new(segment_capacity);

        const SIZE: u16 = 20;
        for i in 0..SIZE {
            segment_list.push(i);
        }
        for i in 0..SIZE {
            let expected = (i + 1) % SIZE;
            assert_eq!(segment_list.next(i), expected);

            let expected = (i + SIZE - 1) % SIZE;
            assert_eq!(segment_list.prev(i), expected);
        }

        // insert_next
        segment_list.insert_next(SIZE, 10);
        assert_eq!(segment_list.prev(10), 9);
        assert_eq!(segment_list.next(10), SIZE);
        assert_eq!(segment_list.prev(SIZE), 10);
        assert_eq!(segment_list.next(SIZE), 11);
        assert_eq!(segment_list.prev(11), SIZE);
        assert_eq!(segment_list.next(11), 12);

        // remove
        segment_list.remove(SIZE);
        for i in 0..SIZE {
            let expected = (i + 1) % SIZE;
            assert_eq!(segment_list.next(i), expected);

            let expected = (i + SIZE - 1) % SIZE;
            assert_eq!(segment_list.prev(i), expected);
        }

        // insert prev
        segment_list.insert_prev(SIZE, 11);
        assert_eq!(segment_list.prev(10), 9);
        assert_eq!(segment_list.next(10), SIZE);
        assert_eq!(segment_list.prev(SIZE), 10);
        assert_eq!(segment_list.next(SIZE), 11);
        assert_eq!(segment_list.prev(11), SIZE);
        assert_eq!(segment_list.next(11), 12);
    }

    fn check(from: u32, to: u32) {
        const SIZE: usize = 100;
        let solution = ArraySolution::new(SIZE);
        let mut two_level_tree = TwoLeveltreeSolution::<30>::new(&solution);

        for i in 0..SIZE {
            let expected = (i + 1) % SIZE;
            assert_eq!(two_level_tree.next(i as u32), expected as u32);

            let expected = (i + SIZE - 1) % SIZE;
            assert_eq!(two_level_tree.prev(i as u32), expected as u32);
        }

        two_level_tree.swap(from, to);
        two_level_tree.print();

        assert_eq!(two_level_tree.prev(from - 1), from - 2);
        assert_eq!(two_level_tree.next(from - 1), to);
        assert_eq!(two_level_tree.prev(to), from - 1);
        assert_eq!(two_level_tree.next(to), to - 1);
        assert_eq!(two_level_tree.prev(from), from + 1);
        assert_eq!(two_level_tree.next(from), to + 1);
        assert_eq!(two_level_tree.prev(to + 1), from);
        assert_eq!(two_level_tree.next(to + 1), to + 2);
    }

    #[test]
    fn test_two_level_tree_case1() {
        check(37, 94);
    }

    #[test]
    fn test_two_level_tree_case2() {
        check(20, 80);
    }

    #[test]
    fn test_two_level_tree_case3() {
        check(17, 92);
    }

    #[test]
    fn test_two_level_tree_case4() {
        const SIZE: usize = 100;
        let solution = ArraySolution::new(SIZE);
        let mut two_level_tree = TwoLeveltreeSolution::<30>::new(&solution);

        for i in 0..SIZE {
            let expected = (i + 1) % SIZE;
            assert_eq!(two_level_tree.next(i as u32), expected as u32);

            let expected = (i + SIZE - 1) % SIZE;
            assert_eq!(two_level_tree.prev(i as u32), expected as u32);
        }

        // [16, 17, 18, 19] -> [16, 18, 17, 19]
        two_level_tree.swap(17, 18);
        assert_eq!(two_level_tree.prev(16), 15);
        assert_eq!(two_level_tree.next(16), 18);
        assert_eq!(two_level_tree.prev(18), 16);
        assert_eq!(two_level_tree.next(18), 17);
        assert_eq!(two_level_tree.prev(17), 18);
        assert_eq!(two_level_tree.next(17), 19);
        assert_eq!(two_level_tree.prev(19), 17);
        assert_eq!(two_level_tree.next(19), 20);
    }

    #[test]
    fn test_two_level_tree_case5() {
        check(10, 24);
    }

    fn test_sequence(arg_list: Vec<(u32, u32)>) {
        const SIZE: usize = 100;

        let mut solution = ArraySolution::new(SIZE);
        let mut two_level_tree = TwoLeveltreeSolution::<30>::new(&solution);

        for (from, to) in arg_list.into_iter() {
            solution.swap(from, to);
            two_level_tree.swap(from, to);

            // check
            let mut id = 0;
            for _iter in 0..SIZE {
                let next_sol = solution.next(id);
                let next_two_level_tree = two_level_tree.next(id);
                assert_eq!(next_sol, next_two_level_tree);
                assert_eq!(solution.prev(id), two_level_tree.prev(id));
                id = next_sol;
            }
        }
    }

    #[test]
    fn test_two_level_tree_case6() {
        // [0, ..., 99]
        // -> [0, 99, 98, 97, ..., 91, 48, 49, ..., 90, 47, 46, ..., 11, 10, 9, ..., 2, 1]
        // -> [0, 1, 2, ..., 9, 10, 97, ..., 91, 48, 49, ..., 90, 47, ..., 11, 98, 99]
        test_sequence(vec![(91, 47), (10, 98)]);
    }

    #[test]
    fn test_two_level_tree_case7() {
        test_sequence(vec![(35, 37), (7, 37)]);
    }

    #[test]
    fn test_two_level_tree_case8() {
        check(58, 18);
    }

    #[test]
    fn test_two_level_tree_case9() {
        const SIZE: usize = 100;
        let solution = ArraySolution::new(SIZE);
        let mut two_level_tree = TwoLeveltreeSolution::<30>::new(&solution);

        for i in 0..SIZE {
            let expected = (i + 1) % SIZE;
            assert_eq!(two_level_tree.next(i as u32), expected as u32);

            let expected = (i + SIZE - 1) % SIZE;
            assert_eq!(two_level_tree.prev(i as u32), expected as u32);
        }

        // [..., 27, 28, 29, 30, ...]
        // -> [..., 30, 29, 28, 27, ...]
        two_level_tree.swap(29, 28);
        assert_eq!(two_level_tree.prev(30), 31);
        assert_eq!(two_level_tree.next(30), 29);
        assert_eq!(two_level_tree.prev(29), 30);
        assert_eq!(two_level_tree.next(29), 28);
        assert_eq!(two_level_tree.prev(28), 29);
        assert_eq!(two_level_tree.next(28), 27);
        assert_eq!(two_level_tree.prev(27), 28);
        assert_eq!(two_level_tree.next(27), 26);
    }

    #[test]
    fn test_two_level_tree_case10() {
        test_sequence(vec![
            (7, 94),
            (64, 17),
            (51, 95),
            (48, 47),
            (51, 79),
            (29, 30),
        ]);
    }

    #[test]
    fn test_two_level_tree_case11() {
        test_sequence(vec![(46, 83), (99, 3)])
    }

    #[test]
    fn test_two_level_tree_case12() {
        test_sequence(vec![(21, 1), (68, 82), (83, 53), (9, 51), (29, 24)]);
    }

    #[test]
    fn test_two_level_tree_random() {
        const SIZE: usize = 100;

        let mut solution = ArraySolution::new(SIZE);
        let mut two_level_tree = TwoLeveltreeSolution::<30>::new(&solution);

        let mut rng = rand::thread_rng();
        let mut max_content = 0;

        for _iter in 0..10_000 {
            if max_content < two_level_tree.segment_list.content.len() as u16 {
                eprintln!("used: {}", two_level_tree.segment_list.content.len());
                max_content = two_level_tree.segment_list.content.len() as u16;
            }
            let from = rng.gen_range(0..SIZE as u32);
            let to = rng.gen_range(0..SIZE as u32);
            if from != to {
                solution.swap(from, to);
                two_level_tree.swap(from, to);

                // check
                let mut id = 0;
                for _iter in 0..SIZE {
                    let next_sol = solution.next(id);
                    let next_two_level_tree = two_level_tree.next(id);
                    assert_eq!(next_sol, next_two_level_tree);
                    assert_eq!(solution.prev(id), two_level_tree.prev(id));
                    id = next_sol;
                }
            }
        }
    }

    #[test]
    fn test_between() {
        const SIZE: usize = 100;
        let solution = ArraySolution::new(SIZE);
        let two_level_tree = TwoLeveltreeSolution::<30>::new(&solution);

        assert!(two_level_tree.between(3, 1, 5));
        assert!(!two_level_tree.between(1, 3, 5));
        assert!(!two_level_tree.between(1, 3, 30));
        assert!(two_level_tree.between(80, 50, 20));
        assert!(two_level_tree.between(59, 55, 20));
        assert!(two_level_tree.between(15, 55, 20));
    }

    #[test]
    fn test_between2() {
        const SIZE: usize = 100;
        let solution = ArraySolution::new(SIZE);
        let two_level_tree = TwoLeveltreeSolution::<30>::new(&solution);

        for i in 0..SIZE as u32 {
            for j in 0..SIZE as u32 {
                for k in 0..SIZE as u32 {
                    assert_eq!(solution.between(i, j, k), two_level_tree.between(i, j, k));
                }
            }
        }
    }

    #[test]
    fn test_between3() {
        const SIZE: usize = 101;
        let mut solution = ArraySolution::new(SIZE);
        let mut two_level_tree = TwoLeveltreeSolution::<30>::new(&solution);

        for iter in 1..=SIZE {
            for i in 0..SIZE as u32 {
                for j in 0..SIZE as u32 {
                    for k in 0..SIZE as u32 {
                        assert_eq!(solution.between(i, j, k), two_level_tree.between(i, j, k));
                    }
                }
            }
            let from = ((iter * 12434121) % SIZE) as u32;
            let to = ((iter * 121343425) % (SIZE - 1)) as u32;
            let to = if from == to { to + 1 } else { to };
            solution.swap(from, to);
            two_level_tree.swap(from, to);
        }
    }

    #[test]
    fn test_index_of() {
        // 今回は一致する条件で実験しているだけで、どのような操作を経ても ArraySolution と TwoLevelTree の状態が同じにはならない点に注意
        const SIZE: usize = 101;
        let mut solution = ArraySolution::new(SIZE);
        let mut two_level_tree = TwoLeveltreeSolution::<30>::new(&solution);

        solution.swap(25, 75);
        two_level_tree.swap(25, 75);

        for id in 0..SIZE as u32 {
            assert_eq!(solution.index_of(id), two_level_tree.index_of(id));
        }
    }

    #[test]
    fn test_id_of() {
        const SIZE: usize = 101;
        let solution = ArraySolution::new(SIZE);
        let two_level_tree = TwoLeveltreeSolution::<30>::new(&solution);

        for index in 0..SIZE {
            assert_eq!(solution.id_of(index), two_level_tree.id_of(index));
        }
    }
}
