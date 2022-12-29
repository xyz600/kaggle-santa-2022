pub trait Solution {
    fn prev(&self, id: u32) -> u32;

    fn next(&self, id: u32) -> u32;

    // whether id is in [from, to]
    fn between(&self, id: u32, from: u32, to: u32) -> bool;

    // swap range [from, to]
    fn swap(&mut self, from: u32, to: u32);

    fn len(&self) -> usize;

    fn print(&self) {
        let mut id = 0;
        eprint!("[");
        for _iter in 0..self.len() {
            eprint!("{}, ", id);
            id = self.next(id);
        }
        eprintln!("]");
    }

    fn index_of(&self, id: u32) -> usize;

    fn id_of(&self, index: usize) -> u32;
}
