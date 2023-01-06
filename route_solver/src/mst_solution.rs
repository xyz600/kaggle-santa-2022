use lib::{array_solution::ArraySolution, distance::DistanceFunction};
use rayon::slice::ParallelSliceMut;
use std::collections::HashSet;

struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> UnionFind {
        UnionFind {
            parent: (0..size).collect(),
            size: vec![1; size],
        }
    }

    fn find(&mut self, child: usize) -> usize {
        if self.parent[child] != child {
            let parent = self.find(self.parent[child]);
            self.parent[child] = parent;
        }
        self.parent[child]
    }

    fn unite(&mut self, v1: usize, v2: usize) {
        let p1 = self.find(v1);
        let p2 = self.find(v2);
        if self.size[p1] < self.size[p2] {
            self.parent[p1] = self.parent[p2];
            self.size[p2] += self.size[p1];
        } else {
            self.parent[p2] = self.parent[p1];
            self.size[p1] += self.size[p2];
        }
    }

    fn same(&mut self, v1: usize, v2: usize) -> bool {
        self.find(v1) == self.find(v2)
    }

    fn size(&mut self, v: usize) -> usize {
        let parent = self.find(v);
        self.size[parent]
    }
}

pub fn create_mst_solution(
    distance: &impl DistanceFunction,
    neighbor_list: &Vec<HashSet<u32>>,
) -> ArraySolution {
    let dim = distance.dimension() as usize;
    let mut uf = UnionFind::new(dim);

    let mut distance_list = vec![];
    for from in 0..dim {
        for to in neighbor_list[from].iter() {
            let dist = distance.distance(from as u32, *to);
            distance_list.push((dist, from as u32, *to));
        }
    }
    distance_list.par_sort();

    eprintln!("finish edge list.");

    let mut counter = 0;
    let mut edges = vec![];
    'outer: while counter < dim {
        for (_dist, from, to) in distance_list.iter() {
            let from = *from as usize;
            let to = *to as usize;

            if !uf.same(from, to) {
                uf.unite(from, to);
                edges.push((from, to));
                counter += 1;
                if counter == dim - 1 {
                    break 'outer;
                }
            }
        }
    }

    eprintln!("finish creating mst.");

    // mst から dfs で経路を復元
    // 適当に root を決めて、dfs で潜ってルートを決定
    let mut neighbor_table = vec![vec![]; dim];
    for (from, to) in edges.into_iter() {
        neighbor_table[from].push(to as u32);
        neighbor_table[to].push(from as u32);
    }

    fn inner(
        current: u32,
        visited: &mut Vec<bool>,
        seq: &mut Vec<u32>,
        neighbor_table: &Vec<Vec<u32>>,
    ) {
        seq.push(current);

        for next in neighbor_table[current as usize].iter() {
            if !visited[*next as usize] {
                visited[*next as usize] = true;
                inner(*next, visited, seq, neighbor_table);
            }
        }
    }

    let mut visited = vec![false; dim];
    visited[0] = true;
    let mut seq = vec![];
    inner(0, &mut visited, &mut seq, &neighbor_table);

    eprintln!("finish reconstruct solution");

    ArraySolution::from_array(seq)
}
