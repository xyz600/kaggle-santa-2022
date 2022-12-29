pub trait DistanceFunction {
    // (id1, id2) の距離を返す
    fn distance(&self, id1: u32, id2: u32) -> i64;

    // 次元数を返す
    fn dimension(&self) -> u32;

    fn name(&self) -> String;
}
