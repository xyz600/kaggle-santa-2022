use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    str::FromStr,
};

use lib::{distance::DistanceFunction, solution::Solution};

use crate::{mst_solution::create_mst_solution, onestep_distance::OneStepDistanceFunction, tsp};

struct SubPathDistance {
    coord_set: Vec<(i16, i16)>,
    begin_id: u32,
    end_id: u32,
    orig_distance: OneStepDistanceFunction,
    orig_index_map: HashMap<(i16, i16), u32>,
    name: String,
}

impl SubPathDistance {}

impl DistanceFunction for SubPathDistance {
    fn distance(&self, id1: u32, id2: u32) -> i64 {
        if (self.begin_id == id1 && self.end_id == id2)
            || (self.begin_id == id2 && self.end_id == id1)
        {
            // s-t 間の距離は0に設定
            0
        } else {
            let orig_id1 = self.orig_index_map[&self.coord_set[id1 as usize]];
            let orig_id2 = self.orig_index_map[&self.coord_set[id2 as usize]];
            self.orig_distance.distance(orig_id1, orig_id2)
        }
    }

    fn dimension(&self) -> u32 {
        self.coord_set.len() as u32
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}

fn calculate_subpath(
    coord_list: Vec<(i16, i16)>,
    begin: (i16, i16),
    end: (i16, i16),
    orig_index_map: &HashMap<(i16, i16), u32>,
    name: String,
) -> Vec<(i16, i16)> {
    let orig_distance =
        OneStepDistanceFunction::load(&PathBuf::from_str("data/image.csv").unwrap());

    let mut index_map = HashMap::new();
    for (i, coord) in coord_list.iter().enumerate() {
        index_map.insert(coord, i as u32);
    }
    let index_map = index_map;

    let mut begin_id = std::u32::MAX;
    let mut end_id = std::u32::MAX;
    for i in 0..coord_list.len() {
        if coord_list[i] == begin {
            begin_id = i as u32;
        }
        if coord_list[i] == end {
            end_id = i as u32;
        }
    }
    assert_ne!(begin_id, std::u32::MAX);
    assert_ne!(end_id, std::u32::MAX);
    let begin_id = begin_id;
    let end_id = end_id;

    let distance = SubPathDistance {
        coord_set: coord_list.clone(),
        begin_id,
        end_id,
        orig_distance,
        orig_index_map: orig_index_map.clone(),
        name,
    };

    // ±1のグリッド内部に含まれる近傍で mst
    let mut neighbor_list = vec![HashSet::new(); distance.dimension() as usize];
    for (i, &(y, x)) in coord_list.iter().enumerate() {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let ny = y + dy;
                let nx = x + dx;
                if index_map.contains_key(&(ny, nx)) {
                    let n_index = index_map[&(ny, nx)];
                    neighbor_list[i].insert(n_index);
                }
            }
        }
    }

    let init_solution = create_mst_solution(&distance, &neighbor_list);
    let subpath_solution = tsp::solve_tsp(&distance, init_solution, 1.0 / (255.0 * 10000.0));

    let mut ret = vec![];
    let mut id = begin_id;
    if subpath_solution.prev(begin_id) == end_id {
        for _iter in 0..subpath_solution.len() {
            ret.push(coord_list[id as usize]);
            id = subpath_solution.next(id);
        }
    } else if subpath_solution.next(begin_id) == end_id {
        for _iter in 0..subpath_solution.len() {
            ret.push(coord_list[id as usize]);
            id = subpath_solution.prev(id);
        }
    } else {
        unreachable!();
    }
    ret
}

fn extract_upper_left() -> Vec<(i16, i16)> {
    let mut ret = vec![];
    for y in 64..=128 {
        for x in -128..=0 {
            ret.push((y, x));
        }
    }
    for y in 0..=63 {
        for x in -128..=-1 {
            ret.push((y, x));
        }
    }
    ret
}

fn extract_lower_left() -> Vec<(i16, i16)> {
    let mut ret = vec![];
    ret.push((0, -128));
    for y in -128..=-1 {
        for x in -128..=0 {
            ret.push((y, x));
        }
    }
    ret
}

fn extract_lower_right() -> Vec<(i16, i16)> {
    let mut ret = vec![];
    ret.push((-128, 0));
    for y in -128..=0 {
        for x in 1..=128 {
            ret.push((y, x));
        }
    }
    ret
}

fn extract_upper_right() -> Vec<(i16, i16)> {
    let mut ret = vec![];
    for y in 64..=128 {
        for x in 1..=128 {
            ret.push((y, x));
        }
    }
    for y in 1..=63 {
        for x in 2..=128 {
            ret.push((y, x));
        }
    }
    ret
}

// subset に分けて、s-t パスを繋いで最適化する
// 適切な配置から開始すれば、128x128 矩形は割と自由に動けるという理由による
pub fn solve() {
    let mut orig_index_table = HashMap::<(i16, i16), u32>::new();
    {
        let mut index = 0u32;
        for y in (-128..=128).rev() {
            for x in -128..=128 {
                orig_index_table.insert((y, x), index);
                index += 1;
            }
        }
    }
    let orig_index_table = orig_index_table;

    // initialize
    // (0, 0) -> (64, 0)

    // Upper Left
    // (64, 0) -> (0, -128)
    let ul_coord_list = extract_upper_left();
    let ul_subpath = calculate_subpath(
        ul_coord_list,
        (64, 0),
        (0, -128),
        &orig_index_table,
        "subpath_ul".to_string(),
    );

    // Lower Left
    // (0, -128) -> (-128, 0)
    let ll_coord_list = extract_lower_left();
    let ll_subpath = calculate_subpath(
        ll_coord_list,
        (0, -128),
        (-128, 0),
        &orig_index_table,
        "subpath_ll".to_string(),
    );

    // Lower Right
    // (-128, 0) -> (0, 128)
    let lr_coord_list = extract_lower_right();
    let lr_subpath = calculate_subpath(
        lr_coord_list,
        (-128, 0),
        (0, 128),
        &orig_index_table,
        "subpath_lr".to_string(),
    );

    // Upper Right
    // (0, 128) -> (64, 1)
    let ur_coord_list = extract_upper_right();
    let ur_subpath = calculate_subpath(
        ur_coord_list,
        (0, 128),
        (64, 1),
        &orig_index_table,
        "subpath_ur".to_string(),
    );

    // finalize
    // (64, 1) -> (1, 1) -> (0, 0)

    // merge solution
}
