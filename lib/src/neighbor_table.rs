use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

use crate::distance::DistanceFunction;

use proconio::input;
use proconio::source::auto::AutoSource;
use rayon::prelude::*;
use std::io::Read;

macro_rules! input_fromfile {
    (path: $path:expr, $($t:tt)+) => {
        fn read_all(filepath: &PathBuf) -> String {
            let mut f = File::open(filepath).expect("file not found");
            let mut contents = String::new();

            f.read_to_string(&mut contents)
                .expect("something went wrong reading the file");

            contents
        }
        let contents = read_all($path);
        let source = AutoSource::from(contents.as_str());

        input! {
            from source,
            $($t)*
        }
    };
}

pub struct NeighborTable {
    table: Vec<Vec<u32>>,
}

impl NeighborTable {
    pub fn new(
        distance: &(impl DistanceFunction + std::marker::Sync),
        neighbor_size: usize,
        create_parallel: bool,
    ) -> NeighborTable {
        let n = distance.dimension();
        let table = if create_parallel {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut distance_list = vec![];
                    for j in 0..n {
                        if i != j {
                            distance_list.push((distance.distance(i, j), j));
                        }
                    }
                    distance_list.sort();
                    distance_list
                        .iter()
                        .take(neighbor_size)
                        .map(|(_, index)| *index)
                        .collect()
                })
                .collect()
        } else {
            (0..n)
                .map(|i| {
                    let mut distance_list = vec![];
                    for j in 0..n {
                        if i != j {
                            distance_list.push((distance.distance(i, j), j));
                        }
                    }
                    distance_list.sort();
                    distance_list
                        .iter()
                        .take(neighbor_size)
                        .map(|(_, index)| *index)
                        .collect()
                })
                .collect()
        };
        NeighborTable { table }
    }

    pub fn neighbor_list(&self, id: u32) -> &Vec<u32> {
        &self.table[id as usize]
    }

    pub fn save(&self, filepath: &PathBuf) {
        let f = File::create(filepath).unwrap();
        let mut writer = BufWriter::new(f);

        writer
            .write(format!("{} {}\n", self.table.len(), self.table[0].len()).as_bytes())
            .unwrap();
        for row in self.table.iter() {
            let line = row.iter().map(|n| n.to_string()).collect::<Vec<_>>();
            writer.write(line.join(" ").as_bytes()).unwrap();
            writer.write("\n".as_bytes()).unwrap();
        }
    }

    pub fn load(filepath: &PathBuf) -> NeighborTable {
        input_fromfile! {
            path: filepath,
            n: usize,
            m: usize,
            table: [[u32; m]; n]
        }

        NeighborTable { table }
    }
}
