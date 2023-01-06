use proconio::input;
use proconio::source::auto::AutoSource;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

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

#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub x: i64,
    pub y: i64,
    pub r: i64,
    pub g: i64,
    pub b: i64,
}

pub fn load_image(filepath: &PathBuf) -> Vec<Cell> {
    const SIZE: usize = 257;
    const LEN: usize = SIZE * SIZE;

    input_fromfile! {
        path: filepath,
        _1: String,
        _2: String,
        _3: String,
        _4: String,
        _5: String,
        data: [(i64, i64, f64, f64, f64); LEN],
    };
    data.iter()
        .map(|(x, y, r, g, b)| Cell {
            x: *x,
            y: *y,
            r: (*r * 255.0).round() as i64,
            g: (*g * 255.0).round() as i64,
            b: (*b * 255.0).round() as i64,
        })
        .collect::<Vec<_>>()
}
