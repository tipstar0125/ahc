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
use rand::{prelude::*, SeedableRng};
use rand_chacha::ChaCha20Rng;
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
            height.chmax(row + 1);
            width.chmax(col + 1);
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

    let mut turn = 0;
    let mut cost = 0.0_f64;
    let mut ac = "WA";

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);

    if M <= 20 {
        let PATTERN_MAX = 1e7 as u128;
        let PATTERN_LIMIT = 1e30 as u128;
        let mut mp = DynamicMap2d::new(vec![None; N * N], N);
        #[allow(unused_assignments)]
        let mut last_coord_cands = vec![];
        let mut before_pattern_num = 1_u128 << 60;
        loop {
            let mut coord_cands = vec![vec![]; M];
            let mut vote = DynamicMap2d::new(vec![vec![]; N * N], N);

            for (m, mino) in minos.iter().enumerate() {
                for i in 0..N - mino.height + 1 {
                    'outer: for j in 0..N - mino.width + 1 {
                        let pos = Coord::new(i, j);
                        for adj in mino.coord_diff.iter() {
                            let next = pos + *adj;
                            if let Some(x) = mp[next] {
                                if x == 0 {
                                    continue 'outer;
                                }
                            }
                        }
                        coord_cands[m].push(pos);
                        for adj in mino.coord_diff.iter() {
                            let next = pos + *adj;
                            vote[next].push((m, (i, j)));
                        }
                    }
                }
            }

            let mut pattern_num_vec = vec![];
            let mut pattern_num = 1_u128;
            for coords in coord_cands.iter() {
                pattern_num_vec.push(coords.len());
                if pattern_num < PATTERN_LIMIT {
                    pattern_num *= coords.len() as u128;
                }
            }

            if pattern_num < PATTERN_MAX {
                last_coord_cands = coord_cands;
                break;
            }
            if pattern_num == before_pattern_num {
                eprintln!("Same num");
                last_coord_cands = coord_cands;
                break;
            }
            before_pattern_num = pattern_num;
            eprintln!("{}", pattern_num);

            let mut search_cands = vec![];
            for i in 0..N {
                for j in 0..N {
                    let pos = Coord::new(i, j);
                    if vote[pos].is_empty() {
                        continue;
                    }
                    let mut delete_set = vec![HashSet::new(); M];
                    for (m, st) in vote[pos].iter() {
                        delete_set[*m].insert(st);
                    }
                    let mut num = 1;
                    for m in 0..M {
                        if num < PATTERN_LIMIT {
                            num *= pattern_num_vec[m] as u128 - delete_set[m].len() as u128;
                        }
                    }
                    search_cands.push((num, (i, j)));
                }
            }
            search_cands.sort();
            search_cands.reverse();
            while let Some((_, (i, j))) = search_cands.pop() {
                let search_coord = Coord::new(i, j);
                if mp[search_coord].is_some() {
                    continue;
                }
                let out = query(&[search_coord], &ans, eps, &es, &mut turn, &mut cost);
                make_query(&[search_coord]);
                // input! {out:u8};
                mp[search_coord] = Some(out);
                if out == 0 {
                    break;
                }
            }
        }

        let mut Q = VecDeque::new();
        let a = DynamicMap2d::new(vec![0_u8; N * N], N);
        let mut ans_cands = vec![];
        Q.push_back((0, a));
        'outer1: while let Some((pos, a)) = Q.pop_front() {
            if pos == M {
                for i in 0..N {
                    for j in 0..N {
                        let u = Coord::new(i, j);
                        if let Some(x) = mp[u] {
                            if x != a[u] {
                                continue 'outer1;
                            }
                        }
                    }
                }

                ans_cands.push(a);
                continue;
            }
            'outer2: for st in last_coord_cands[pos].iter() {
                let mut na = a.clone();
                for adj in minos[pos].coord_diff.iter() {
                    let next = *st + *adj;
                    na[next] += 1;
                    if mp[next].is_some() && na[next] > mp[next].unwrap() {
                        continue 'outer2;
                    }
                }
                Q.push_back((pos + 1, na));
            }
        }

        while ans_cands.len() > 10 {
            let mut search_coord_cands = vec![];
            for i in 0..N {
                for j in 0..N {
                    let pos = Coord::new(i, j);
                    if mp[pos].is_some() {
                        continue;
                    }
                    search_coord_cands.push(pos);
                }
            }
            search_coord_cands.shuffle(&mut rng);

            while let Some(pos) = search_coord_cands.pop() {
                if mp[pos].is_some() {
                    continue;
                }
                let out = query(&[pos], &ans, eps, &es, &mut turn, &mut cost);
                make_query(&[pos]);
                // input! {out:u8};
                mp[pos] = Some(out);
                break;
            }

            let mut next_ans_cands = vec![];
            'outer: for a in ans_cands.iter() {
                for i in 0..N {
                    for j in 0..N {
                        let u = Coord::new(i, j);
                        if let Some(x) = mp[u] {
                            if x != a[u] {
                                continue 'outer;
                            }
                        }
                    }
                }
                next_ans_cands.push(a.clone());
            }
            ans_cands = next_ans_cands;
        }

        for a in ans_cands.iter() {
            turn += 1;
            make_answer(a);
            if check_ans(a, &ans) {
                ac = "AC";
                break;
            }
            // input! {out:usize}
            // if out == 1 {
            //     break;
            // }
            cost += 1.0;
        }
    } else {
        let mut mino_num = 0;
        for mino in minos.iter() {
            mino_num += mino.coord_diff.len();
        }

        let mut cnt = 0;
        let mut mp = DynamicMap2d::new(vec![None; N * N], N);

        'outer: while cnt < mino_num {
            let mut i = rng.gen_range(0..N);
            let mut j = rng.gen_range(0..N);
            loop {
                if mp[Coord::new(i, j)].is_none() {
                    break;
                }
                i = rng.gen_range(0..N);
                j = rng.gen_range(0..N);
            }
            let now = Coord::new(i, j);

            let out = query(&[now], &ans, eps, &es, &mut turn, &mut cost); // Local
            make_query(&[now]);
            // input! {out:u8}

            if out == 0 {
                mp[now] = Some(0);
                continue;
            }
            cnt += out as usize;
            if cnt >= mino_num {
                break;
            }
            mp[now] = Some(out);
            let mut Q = VecDeque::new();
            Q.push_back(now);

            while let Some(u) = Q.pop_front() {
                for adj in ADJ {
                    let v = u + adj;
                    if !v.in_map(N, N) {
                        continue;
                    }
                    if mp[v].is_some() {
                        continue;
                    }
                    let out = query(&[v], &ans, eps, &es, &mut turn, &mut cost); // Local
                    make_query(&[v]);
                    // input! {out:u8}

                    if out == 0 {
                        mp[v] = Some(0);
                    } else {
                        mp[v] = Some(out);
                        Q.push_back(v);
                        cnt += out as usize;
                        if cnt >= mino_num {
                            break 'outer;
                        }
                    }
                }
            }
        }

        let mut a = DynamicMap2d::new(vec![0; N * N], N);
        for i in 0..N {
            for j in 0..N {
                let pos = Coord::new(i, j);
                if let Some(x) = mp[pos] {
                    a[pos] = x;
                }
            }
        }

        make_answer(&a);
        if check_ans(&a, &ans) {
            ac = "AC";
        }
        // input! {out:u8}
    }
    let score = (1e6 * cost.max(1.0 / N as f64)).round() as usize;
    eprintln!("Cost: {}", cost / (N as f64 * N as f64));
    // eprintln!("Cost: {}", cost);
    eprintln!("Score: {score}");

    #[allow(unused_mut, unused_assignments)]
    let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
    #[cfg(feature = "local")]
    {
        eprintln!("Local Mode");
        elapsed_time *= 0.55;
    }
    eprintln!("Elapsed: {}", (elapsed_time * 1000.0) as usize);
    eprintln!("Check: {ac}");
}

fn calc_prob(mut out: f64, k: usize, eps: f64) -> Vec<f64> {
    let base = 3;
    let num = (base - 1) * k;
    let mut probs = vec![0.0; num + 1];
    let std = (k as f64 * eps * (1.0 - eps)).sqrt();
    let delta = std * 0.1;
    if out == 0.0 {
        out -= std;
    }
    for i in 0..=num {
        let mean = (k as f64 - i as f64) * eps + (i as f64) * (1.0 - eps);
        let diff = (mean - out).abs();
        probs[i] = e(diff + delta, std) - e(diff - delta, std);
    }

    let sum = probs.iter().sum::<f64>().min(1.0);
    if sum == 0.0 {
        probs = vec![0.0; num + 1];
        probs[num] = 1.0;
    } else {
        let sum_inv = 1.0 / sum;
        for i in 0..=num {
            probs[i] *= sum_inv;
        }
    }

    let pattern_num = base.pow(k as u32);
    let mut cnt_pattern = vec![vec![]; num + 1];
    for i in 0..pattern_num {
        let mut n = i;
        let mut s = 0;
        while n > 0 {
            s += n % base;
            n /= base;
        }
        cnt_pattern[s].push(i);
    }
    let mut prob = vec![0.0; base];
    for i in 0..=num {
        for p in cnt_pattern[i].iter() {
            prob[p % base] += probs[i] / cnt_pattern[i].len() as f64;
        }
    }
    prob
}

fn e(x: f64, std: f64) -> f64 {
    0.5 + 0.5 * erf(x / std / 2.0_f64.sqrt())
}

fn erf(x: f64) -> f64 {
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let t = 1.0 / (1.0 + P * x);
    1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp()
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

fn make_answer(mp: &DynamicMap2d<u8>) {
    let N = mp.size;
    let mut coords = vec![];
    for i in 0..N {
        for j in 0..N {
            let coord = Coord::new(i, j);
            if mp[coord] > 0 {
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

fn check_ans(a: &DynamicMap2d<u8>, ans: &DynamicMap2d<i32>) -> bool {
    let N = a.size;
    for i in 0..N {
        for j in 0..N {
            let pos = Coord::new(i, j);
            if ans[pos] as u8 != a[pos] {
                return false;
            }
        }
    }
    true
}

#[derive(Debug, Clone, Copy)]
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

pub const ADJACENT4: [CoordDiff; 4] = [
    CoordDiff::new(0, 0),
    CoordDiff::new(0, 1),
    CoordDiff::new(1, 0),
    CoordDiff::new(1, 1),
];

pub const ADJACENT6: [CoordDiff; 6] = [
    CoordDiff::new(0, 0),
    CoordDiff::new(0, 1),
    CoordDiff::new(1, 0),
    CoordDiff::new(1, 1),
    CoordDiff::new(2, 0),
    CoordDiff::new(2, 1),
];

pub const ADJACENT8: [CoordDiff; 8] = [
    CoordDiff::new(0, 0),
    CoordDiff::new(0, 1),
    CoordDiff::new(1, 0),
    CoordDiff::new(1, 1),
    CoordDiff::new(2, 0),
    CoordDiff::new(2, 1),
    CoordDiff::new(3, 0),
    CoordDiff::new(3, 1),
];

pub const ADJACENT10: [CoordDiff; 10] = [
    CoordDiff::new(0, 0),
    CoordDiff::new(0, 1),
    CoordDiff::new(1, 0),
    CoordDiff::new(1, 1),
    CoordDiff::new(2, 0),
    CoordDiff::new(2, 1),
    CoordDiff::new(3, 0),
    CoordDiff::new(3, 1),
    CoordDiff::new(4, 0),
    CoordDiff::new(4, 1),
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

pub trait ChangeMinMax {
    fn chmin(&mut self, x: Self) -> bool;
    fn chmax(&mut self, x: Self) -> bool;
}

impl<T: PartialOrd> ChangeMinMax for T {
    fn chmin(&mut self, x: Self) -> bool {
        *self > x && {
            *self = x;
            true
        }
    }
    fn chmax(&mut self, x: Self) -> bool {
        *self < x && {
            *self = x;
            true
        }
    }
}

#[derive(Debug, Clone)]
struct TimeKeeper {
    start_time: std::time::Instant,
    time_threshold: f64,
}

impl TimeKeeper {
    fn new(time_threshold: f64) -> Self {
        TimeKeeper {
            start_time: std::time::Instant::now(),
            time_threshold,
        }
    }
    #[inline]
    fn isTimeOver(&self) -> bool {
        let elapsed_time = self.start_time.elapsed().as_nanos() as f64 * 1e-9;
        #[cfg(feature = "local")]
        {
            elapsed_time * 0.55 >= self.time_threshold
        }
        #[cfg(not(feature = "local"))]
        {
            elapsed_time >= self.time_threshold
        }
    }
    #[inline]
    fn get_time(&self) -> f64 {
        let elapsed_time = self.start_time.elapsed().as_nanos() as f64 * 1e-9;
        #[cfg(feature = "local")]
        {
            elapsed_time * 0.55
        }
        #[cfg(not(feature = "local"))]
        {
            elapsed_time
        }
    }
}
