#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::neg_multiply)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet, BinaryHeap, HashSet, VecDeque},
};

use itertools::Itertools;
use proconio::{
    input,
    marker::{Chars, Usize1},
};
use rand::prelude::*;
use superslice::Ext;

fn main() {
    let start = std::time::Instant::now();

    let mut stdin =
        proconio::source::line::LineSource::new(std::io::BufReader::new(std::io::stdin()));
    macro_rules! input(($($tt:tt)*) => (proconio::input!(from &mut stdin, $($tt)*)));

    input! {
        N: usize,
        M: usize,
        eps: f64,
    }

    let mut minos = vec![];
    for _ in 0..M {
        input! {
            d: usize,
            coords: [(usize, usize); d]
        }
        let mut coord_diff = vec![];
        let mut height = 0;
        let mut width = 0;
        for coord in coords.iter() {
            let row = coord.0;
            let col = coord.1;
            height = height.max(row + 1);
            width = width.max(col + 1);
            coord_diff.push(CoordDiff::new(row as isize, col as isize));
        }
        let mino = Mino {
            coord_diff,
            height,
            width,
        };
        minos.push(mino);
    }

    // Local
    input! {
        _ps: [(usize, usize); M],
        ans: [[i32; N]; N],
        es: [f64; 2*N*N]
    }
    let ans = ans.into_iter().flatten().collect_vec();
    let ans = DynamicMap2d::new(ans, N);

    let candidate_mino_coords = make_candidate_mino_coords(N, &minos);
    let boards = make_boards(N, &minos, &candidate_mino_coords);

    let mut turn = 0;
    let mut cost = 0.0_f64;
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);

    let board_num = boards.len();
    let mut prob = vec![1.0 / (board_num as f64); board_num];

    loop {
        let coords = decide_fortune(N, eps, &mut rng, &prob, &boards);
        let k = coords.len();
        let ret = query(&coords, &ans, eps, &es, &mut turn, &mut cost);
        make_query(&coords);
        // input! {ret:u8};
        for i in 0..board_num {
            let mut cnt = 0;
            for coord in coords.iter() {
                cnt += boards[i][*coord];
            }
            prob[i] *= likelihood(k, eps, cnt, ret);
        }
        normalize(&mut prob);
        let mut mx = 0.0;
        let mut idx = 0;
        for (i, &p) in prob.iter().enumerate() {
            if mx < p {
                mx = p;
                idx = i;
            }
        }
        if mx > 0.8 {
            make_answer(&boards[idx]);
            let ret = eval(&boards[idx], &ans);
            // input! {ret:u8};
            if ret == 1 {
                break;
            }
            cost += 1.0;
            prob[idx] = 0.0;
        }
    }

    let score = (1e6 * cost.max(1.0 / N as f64)).round() as usize;
    eprintln!("Turn: {}", turn as f64 / (2.0 * N as f64 * N as f64));
    eprintln!("Cost: {}", cost);
    eprintln!("Score: {score}");

    #[allow(unused_mut, unused_assignments)]
    let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
    #[cfg(feature = "local")]
    {
        eprintln!("Local Mode");
        elapsed_time *= 0.55;
    }
    eprintln!("Elapsed: {}", (elapsed_time * 1000.0) as usize);
}

fn make_candidate_mino_coords(N: usize, minos: &[Mino]) -> Vec<Vec<Coord>> {
    let mut ret = vec![];
    for mino in minos.iter() {
        let mut cands = vec![];
        for i in 0..N - mino.height + 1 {
            for j in 0..N - mino.width + 1 {
                let pos = Coord::new(i, j);
                cands.push(pos);
            }
        }
        ret.push(cands);
    }
    ret
}

fn make_boards(
    N: usize,
    minos: &[Mino],
    candidate_mino_coords: &[Vec<Coord>],
) -> Vec<DynamicMap2d<u8>> {
    let M = minos.len();
    let mut ret = vec![];
    let mut Q = VecDeque::new();
    Q.push_back((0, DynamicMap2d::new(vec![0_u8; N * N], N)));
    while let Some((cnt, B)) = Q.pop_front() {
        if cnt == M {
            ret.push(B.clone());
            continue;
        }
        for &pos in candidate_mino_coords[cnt].iter() {
            let mut nB = B.clone();
            for &coord_diff in minos[cnt].coord_diff.iter() {
                let nxt = pos + coord_diff;
                nB[nxt] += 1;
            }
            Q.push_back((cnt + 1, nB));
        }
    }
    ret
}

fn decide_fortune(
    N: usize,
    eps: f64,
    rng: &mut rand_chacha::ChaCha20Rng,
    prob: &[f64],
    boards: &[DynamicMap2d<u8>],
) -> Vec<Coord> {
    let mut fortune_coords = HashSet::new();
    for i in 0..N {
        for j in 0..N {
            fortune_coords.insert(Coord::new(i, j));
        }
    }
    let mut best_score = 0.0;
    let mut best_fortune_coords = fortune_coords.clone();
    let mut k = N * N;

    let L = boards.len();
    let mut cnt = vec![0; L];
    for (i, board) in boards.iter().enumerate() {
        for coord in fortune_coords.iter() {
            cnt[i] += board[*coord];
        }
    }

    for _ in 0..1000 {
        let i = rng.gen_range(0..N);
        let j = rng.gen_range(0..N);
        let coord = Coord::new(i, j);
        let mut is_removed = false;
        if fortune_coords.contains(&coord) {
            fortune_coords.remove(&coord);
            for (i, board) in boards.iter().enumerate() {
                cnt[i] -= board[coord];
            }
            k -= 1;
            is_removed = true;
        } else {
            fortune_coords.insert(coord);
            for (i, board) in boards.iter().enumerate() {
                cnt[i] += board[coord];
            }
            k += 1;
        }

        let score = calc_mutual_information(k, eps, prob, &cnt);
        if score > best_score {
            best_score = score;
            best_fortune_coords = fortune_coords.clone();
        } else if is_removed {
            fortune_coords.insert(coord);
            for (i, board) in boards.iter().enumerate() {
                cnt[i] += board[coord];
            }
            k += 1;
        } else {
            fortune_coords.remove(&coord);
            for (i, board) in boards.iter().enumerate() {
                cnt[i] -= board[coord];
            }
            k -= 1;
        }
    }
    best_fortune_coords.into_iter().collect()
}

fn calc_mutual_information(k: usize, eps: f64, prob: &[f64], cnt: &[u8]) -> f64 {
    let query_result_num = 40;
    let mut query_result_p_vec = vec![0.0; query_result_num + 1];

    for (&p, &c) in prob.iter().zip(cnt) {
        for query_result in 0..=query_result_num {
            query_result_p_vec[query_result] += p * likelihood(k, eps, c, query_result as u8);
        }
    }
    let mut ret = 0.0;

    for (query_result, query_result_p) in query_result_p_vec.iter().enumerate() {
        if *query_result_p == 0.0 {
            continue;
        }
        for (&p, &c) in prob.iter().zip(cnt) {
            let likelihood = likelihood(k, eps, c, query_result as u8);
            if likelihood == 0.0 {
                continue;
            }
            ret += likelihood * p * (likelihood / query_result_p).log2();
        }
    }

    ret * (k as f64).sqrt()
}

#[allow(clippy::approx_constant)]
fn normal_cdf(x: f64, mean: f64, std_dev: f64) -> f64 {
    0.5 * (1.0 + libm::erf((x - mean) / (std_dev * 1.41421356237)))
}

fn probability_in_range(mean: f64, std_dev: f64, a: f64, b: f64) -> f64 {
    if mean < a {
        return probability_in_range(mean, std_dev, 2.0 * mean - b, 2.0 * mean - a);
    }
    let p_a = normal_cdf(a, mean, std_dev);
    let p_b = normal_cdf(b, mean, std_dev);
    p_b - p_a
}

fn likelihood(k: usize, eps: f64, cnt: u8, res: u8) -> f64 {
    let mean = (k as f64 - cnt as f64) * eps + (cnt as f64) * (1.0 - eps);
    let std_dev = ((k as f64) * eps * (1.0 - eps)).sqrt();
    if res == 0 {
        probability_in_range(mean, std_dev, -1e10, res as f64 + 0.5)
    } else {
        probability_in_range(mean, std_dev, res as f64 - 0.5, res as f64 + 0.5)
    }
}

fn normalize(prob: &mut [f64]) {
    let s = prob.iter().sum::<f64>();
    assert!(s >= 0.0);
    for p in prob.iter_mut() {
        *p /= s;
    }
}

#[derive(Debug)]
struct Mino {
    coord_diff: Vec<CoordDiff>,
    height: usize,
    width: usize,
}

fn query(
    coords: &[Coord],
    ans: &DynamicMap2d<i32>,
    eps: f64,
    es: &[f64],
    turn: &mut usize,
    cost: &mut f64,
) -> u8 {
    let k = coords.len() as f64;
    *turn += 1;
    *cost += 1.0 / k.sqrt();
    if k == 1.0 {
        return ans[coords[0]] as u8;
    }
    let mut vs = 0.0;
    for coord in coords.iter() {
        vs += ans[*coord] as f64;
    }
    let mean = (k - vs) * eps + vs * (1.0 - eps);
    let std = (k * eps * (1.0 - eps)).sqrt();
    let ret = mean + std * es[*turn];
    let ret = ret.round() as usize;
    ret.max(0) as u8
}

fn make_query(coords: &[Coord]) {
    let mut v = vec![];
    v.push(coords.len());
    for coord in coords.iter() {
        v.push(coord.row);
        v.push(coord.col);
    }
    let mut query = "q ".to_string();
    query += v.iter().join(" ").as_str();
    println!("{query}");
}

fn make_answer(a: &DynamicMap2d<u8>) {
    let N = a.size;
    let mut coords = vec![];
    for i in 0..N {
        for j in 0..N {
            let coord = Coord::new(i, j);
            if a[coord] > 0 {
                coords.push(coord);
            }
        }
    }
    let mut v = vec![];
    v.push(coords.len());
    for coord in coords.iter() {
        v.push(coord.row);
        v.push(coord.col);
    }
    let mut ans = "a ".to_string();
    ans += v.iter().join(" ").as_str();
    println!("{ans}");
}

fn eval(a: &DynamicMap2d<u8>, ans: &DynamicMap2d<i32>) -> u8 {
    let N = a.size;
    for i in 0..N {
        for j in 0..N {
            let pos = Coord::new(i, j);
            if ans[pos] as u8 != a[pos] {
                return 0;
            }
        }
    }
    1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coord {
    row: usize,
    col: usize,
}

impl Coord {
    pub fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }
    pub fn in_map(&self, height: usize, width: usize) -> bool {
        self.row < height && self.col < width
    }
    pub fn to_index(&self, width: usize) -> CoordIndex {
        CoordIndex(self.row * width + self.col)
    }
}

impl std::ops::Add<CoordDiff> for Coord {
    type Output = Coord;
    fn add(self, rhs: CoordDiff) -> Self::Output {
        Coord::new(
            self.row.wrapping_add_signed(rhs.dr),
            self.col.wrapping_add_signed(rhs.dc),
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CoordDiff {
    dr: isize,
    dc: isize,
}

impl CoordDiff {
    pub const fn new(dr: isize, dc: isize) -> Self {
        Self { dr, dc }
    }
}

pub const ADJ: [CoordDiff; 4] = [
    CoordDiff::new(1, 0),
    CoordDiff::new(!0, 0),
    CoordDiff::new(0, 1),
    CoordDiff::new(0, !0),
];

pub struct CoordIndex(pub usize);

impl CoordIndex {
    pub fn new(index: usize) -> Self {
        Self(index)
    }
    pub fn to_coord(&self, width: usize) -> Coord {
        Coord {
            row: self.0 / width,
            col: self.0 % width,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DynamicMap2d<T> {
    pub size: usize,
    map: Vec<T>,
}

impl<T> DynamicMap2d<T> {
    pub fn new(map: Vec<T>, size: usize) -> Self {
        assert_eq!(size * size, map.len());
        Self { size, map }
    }
}

impl<T> std::ops::Index<Coord> for DynamicMap2d<T> {
    type Output = T;

    #[inline]
    fn index(&self, coordinate: Coord) -> &Self::Output {
        &self[coordinate.to_index(self.size)]
    }
}

impl<T> std::ops::IndexMut<Coord> for DynamicMap2d<T> {
    #[inline]
    fn index_mut(&mut self, coordinate: Coord) -> &mut Self::Output {
        let size = self.size;
        &mut self[coordinate.to_index(size)]
    }
}

impl<T> std::ops::Index<CoordIndex> for DynamicMap2d<T> {
    type Output = T;

    fn index(&self, index: CoordIndex) -> &Self::Output {
        unsafe { self.map.get_unchecked(index.0) }
    }
}

impl<T> std::ops::IndexMut<CoordIndex> for DynamicMap2d<T> {
    #[inline]
    fn index_mut(&mut self, index: CoordIndex) -> &mut Self::Output {
        unsafe { self.map.get_unchecked_mut(index.0) }
    }
}
