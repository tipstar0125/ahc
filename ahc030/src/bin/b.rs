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
    collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque},
};

use itertools::Itertools;
use proconio::{
    input,
    marker::{Chars, Usize1},
};
use rand::{prelude::*, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, Normal};
use superslice::Ext;
use svg::node::element::path::Data;
use svg::node::element::{Circle, Group, Line, Path, Rectangle, Style, Text, Title};
use svg::node::Text as TextContent;
use svg::Document;

fn main() {
    // input! {
    //     N: usize,
    //     M: usize,
    //     eps: f64,
    // }

    // input! {
    //     ps: [(usize, usize); M],
    //     ans: [[usize; N]; N],
    //     es: [f64; 2*N*N]
    // }
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(10);
    for _ in 0..10 {
        let N = 30;
        let M = 4;
        let eps = 0.01;
        let input = gen(10, Some(N), Some(M), Some(eps), &mut rng);
        let ans = input.ans;
        let ps = input.ps;
        let es = input.es;

        let mut minos = vec![];
        for i in 0..M {
            // input! {
            //     d: usize,
            //     coords: [(usize, usize); d]
            // }
            let coords = input.ts[i].clone();
            let mut coords2 = vec![];
            let mut height = 0;
            let mut width = 0;
            for coord in coords.iter() {
                let row = coord.0;
                let col = coord.1;
                height.chmax(row + 1);
                width.chmax(col + 1);
                coords2.push(Coord::new(row, col));
            }
            let mino = Mino {
                coords: coords2,
                height,
                width,
            };
            minos.push(mino);
        }

        let ans = ans.into_iter().flatten().collect_vec();
        let ans = DynamicMap2d::new(ans, N);
        let mut turn = 0;

        let mut mining = DynamicMap2d::new(vec![0.0; N * N], N);
        let mut opacity = DynamicMap2d::new(vec![0.0; N * N], N);

        for i in 0..N {
            for j in 0..N {
                let now = Coord::new(i, j);
                let mut coords = vec![];
                for adj in ADJACENT {
                    let mut next = now + adj;
                    if !next.in_map(N, N) {
                        continue;
                        // let ni = next.row % N;
                        // let nj = next.col % N;
                        // next = Coord::new(ni, nj);
                    }
                    coords.push(next);
                }
                let ret = query(&coords, &ans, eps, &es, &mut turn);
                for coord in coords.iter() {
                    mining[*coord] += ret;
                }
            }
        }
        let mut mx = 0.0;
        for i in 0..N {
            for j in 0..N {
                let now = Coord::new(i, j);
                if mining[now] > mx {
                    mx = mining[now];
                }
            }
        }
        for i in 0..N {
            for j in 0..N {
                let now = Coord::new(i, j);
                opacity[now] = mining[now] / mx;
            }
        }
        vis(&mining, &opacity, &ans, N);

        let mut cands = vec![];
        for mino in minos.iter() {
            let mut cand = vec![];
            for i in 0..N - mino.height + 1 {
                for j in 0..N - mino.width + 1 {
                    let mut score = 0.0;
                    for coord in mino.coords.iter() {
                        score += mining[Coord::new(i + coord.row, j + coord.col)];
                    }
                    cand.push((score.round() as usize, i, j));
                }
            }
            cand.sort();
            cand.reverse();
            cands.push(cand);
        }
        let mut ok = false;
        'outer: for (i, a) in cands[0].iter().take(cands[0].len().min(150)).enumerate() {
            for (j, b) in cands[1].iter().take(cands[1].len().min(150)).enumerate() {
                for (k, c) in cands[2].iter().take(cands[2].len().min(150)).enumerate() {
                    for (l, d) in cands[3].iter().take(cands[3].len().min(150)).enumerate() {
                        let abc = vec![(a.1, a.2), (b.1, b.2), (c.1, c.2), (d.1, d.2)];
                        if abc == ps {
                            ok = true;
                            println!("{} {} {} {}", i, j, k, l);
                            break 'outer;
                        }
                    }
                }
            }
        }
        if ok {
            println!("OK");
        } else {
            println!("NG");
        }
    }
}

#[derive(Debug)]
struct Mino {
    coords: Vec<Coord>,
    height: usize,
    width: usize,
}

fn query(coords: &[Coord], ans: &DynamicMap2d<i32>, eps: f64, es: &[f64], turn: &mut usize) -> f64 {
    let k = coords.len() as f64;
    let mut vs = 0.0;
    for coord in coords.iter() {
        vs += ans[*coord] as f64;
    }
    let mean = (k - vs) * eps + vs * (1.0 - eps);
    let std = (k * eps * (1.0 - eps)).sqrt();
    // let dist = Normal::<f64>::new(mean, std).unwrap();
    // dist.sample(rng).max(0.0)
    let ret = mean + std * es[*turn];
    *turn += 1;
    ret.max(0.0)
}

// fn query_single(
//     // ans: &DynamicMap2d<usize>,
//     eps: f64,
//     rng: &mut ChaCha20Rng,
// ) -> f64 {
//     let k = 9.0;
//     let mut vs = 0.0;
//     let mean = (k - vs) * eps + vs * (1.0 - eps);
//     println!("mean: {}", mean);
//     let std = (k * eps * (1.0 - eps)).sqrt();
//     println!("std: {}", std);
//     let dist = Normal::<f64>::new(mean, std).unwrap();
//     dist.sample(rng).max(0.0)
// }

fn vis(mining: &DynamicMap2d<f64>, opacity: &DynamicMap2d<f64>, ans: &DynamicMap2d<i32>, N: usize) {
    let size = 600.0;
    let mut doc = doc(size, size);
    doc = doc.add(Style::new(format!(
        "text {{text-anchor: middle; dominant-baseline: central; font-size: {}}}",
        12
    )));
    let d = size / N as f32;
    for i in 0..N {
        for j in 0..N {
            let now = Coord::new(i, j);
            let mut rec =
                rect(j as f32 * d, i as f32 * d, d, d, "green").set("fill-opacity", opacity[now]);
            if ans[now] > 0 {
                rec = rec.set("stroke", "red");
            }
            doc = doc.add(rec);
            doc = doc.add(txt(
                j as f32 * d + d / 2.0,
                i as f32 * d + d / 2.0,
                format!("{:.1}", mining[now]).as_str(),
            ));
        }
    }

    let vis = format!("<html><body>{}</body></html>", doc);
    std::fs::write("vis.html", vis).unwrap();
}

#[macro_export]
macro_rules! mat {
	($($e:expr),*) => { Vec::from(vec![$($e),*]) };
	($($e:expr,)*) => { Vec::from(vec![$($e),*]) };
	($e:expr; $d:expr) => { Vec::from(vec![$e; $d]) };
	($e:expr; $d:expr $(; $ds:expr)+) => { Vec::from(vec![mat![$e $(; $ds)*]; $d]) };
}

#[derive(Clone, Debug)]
pub struct Input {
    pub n: usize,
    pub m: usize,
    pub eps: f64,
    pub ts: Vec<Vec<(usize, usize)>>,
    pub ps: Vec<(usize, usize)>,
    pub ans: Vec<Vec<i32>>,
    pub es: Vec<f64>,
}

const DIJ: [(usize, usize); 4] = [(0, 1), (1, 0), (0, !0), (!0, 0)];

fn gen(
    seed: u64,
    fix_N: Option<usize>,
    fix_M: Option<usize>,
    fix_eps: Option<f64>,
    rng: &mut rand_chacha::ChaCha20Rng,
) -> Input {
    // let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
    let mut n = rng.gen_range(10i32..=20) as usize;
    if let Some(v) = fix_N {
        n = v;
    }
    let mut m = rng.gen_range(2i32..=(n * n / 20) as i32) as usize;
    if let Some(v) = fix_M {
        m = v;
    }
    let mut eps = rng.gen_range(1..=20) as f64 / 100.0;
    if let Some(v) = fix_eps {
        eps = v;
    }
    let avg = (rng.gen_range((n * n / 5) as i32..=(n * n / 2) as i32) as usize / m).max(4);
    let delta = rng.gen_range(0..=avg as i32 - 4) as usize;
    let mut ts = vec![];
    for _ in 0..m {
        let size = rng.gen_range((avg - delta) as i32..=(avg + delta) as i32) as usize;
        let mut used = mat![false; n; n];
        let mut list = vec![];
        list.push((n / 2, n / 2));
        used[n / 2][n / 2] = true;
        let mut adj = vec![];
        for (di, dj) in DIJ {
            let i2 = n / 2 + di;
            let j2 = n / 2 + dj;
            adj.push((i2, j2));
            used[i2][j2] = true;
        }
        while list.len() < size {
            let p = rng.gen_range(0..adj.len() as i32) as usize;
            let (i, j) = adj.remove(p);
            list.push((i, j));
            for (di, dj) in DIJ {
                let i2 = i + di;
                let j2 = j + dj;
                if i2 < n && j2 < n && !used[i2][j2] {
                    adj.push((i2, j2));
                    used[i2][j2] = true;
                }
            }
        }
        let min_i = list.iter().map(|x| x.0).min().unwrap();
        let min_j = list.iter().map(|x| x.1).min().unwrap();
        for x in &mut list {
            x.0 -= min_i;
            x.1 -= min_j;
        }
        list.sort();
        ts.push(list);
    }
    let mut ans = mat![0; n; n];
    let mut ps = vec![];
    for p in 0..m {
        let max_i = ts[p].iter().map(|x| x.0).max().unwrap();
        let max_j = ts[p].iter().map(|x| x.1).max().unwrap();
        let di = rng.gen_range(0..(n - max_i) as i32) as usize;
        let dj = rng.gen_range(0..(n - max_j) as i32) as usize;
        for &(i, j) in &ts[p] {
            ans[i + di][j + dj] += 1;
        }
        ps.push((di, dj));
    }
    let mut es = vec![];
    for _ in 0..n * n * 2 {
        es.push(rng.sample(rand_distr::StandardNormal));
    }
    Input {
        n,
        m,
        eps,
        ts,
        ps,
        ans,
        es,
    }
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

pub struct CoordDiff {
    dr: isize,
    dc: isize,
}

impl CoordDiff {
    pub const fn new(dr: isize, dc: isize) -> Self {
        Self { dr, dc }
    }
}

pub const ADJACENT: [CoordDiff; 9] = [
    CoordDiff::new(0, 0),
    CoordDiff::new(0, 1),
    CoordDiff::new(1, 0),
    CoordDiff::new(1, 1),
    CoordDiff::new(0, 2),
    CoordDiff::new(2, 0),
    CoordDiff::new(2, 1),
    CoordDiff::new(1, 2),
    CoordDiff::new(2, 2),
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

const MARGIN: f32 = 10.0;

pub fn doc(height: f32, width: f32) -> Document {
    Document::new()
        .set(
            "viewBox",
            (
                -MARGIN,
                -MARGIN,
                width + 2.0 * MARGIN,
                height + 2.0 * MARGIN,
            ),
        )
        .set("width", width + MARGIN)
        .set("height", height + MARGIN)
        .set("style", "background-color:#F2F3F5")
}

pub fn rect(x: f32, y: f32, w: f32, h: f32, fill: &str) -> Rectangle {
    Rectangle::new()
        .set("x", x)
        .set("y", y)
        .set("width", w)
        .set("height", h)
        .set("fill", fill)
}

pub fn txt(x: f32, y: f32, text: &str) -> Text {
    Text::new()
        .add(TextContent::new(text))
        .set("x", x)
        .set("y", y)
        .set("fill", "black")
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
