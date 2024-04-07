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
    collections::{BinaryHeap, HashMap, VecDeque},
};

use proconio::{fastout, input, marker::Chars};
use rand::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

fn main() {
    let start = std::time::Instant::now();
    let time_limit = 1.8;
    let time_keeper = TimeKeeper::new(time_limit);
    let mut rng = rand_pcg::Pcg64Mcg::new(12345);
    let input = read_input();

    let mut state = State::new(
        &input,
        Coord::new(0, 0),
        Coord::new(input.n - 1, input.n - 1),
        annealing(&input, &mut rng, &time_keeper),
    );

    state.quick_sort(&input);
    state.output();

    #[allow(unused_mut, unused_assignments)]
    let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
    #[cfg(feature = "local")]
    {
        eprintln!("Local Mode");
        elapsed_time *= 0.55;
    }
    eprintln!("Elapsed: {}", (elapsed_time * 1000.0) as usize);
}

fn annealing(
    input: &Input,
    rng: &mut rand_pcg::Pcg64Mcg,
    time_keeper: &TimeKeeper,
) -> DynamicMap2d<i64> {
    let mut state = AnnealingState::new(input.board.clone());
    let mut best_state = state.clone();
    let mut cost = input.cost;
    let mut best_cost = cost;

    let T0 = (4 * input.n.pow(2)) as f64;
    let T1 = 1.0;
    let time_limit = 1.0;

    while time_keeper.get_time() < time_limit {
        let i1 = rng.gen_range(0..input.n);
        let j1 = rng.gen_range(0..input.n);
        let pos1 = Coord::new(i1, j1);
        let i2 = rng.gen_range(0..input.n);
        let j2 = rng.gen_range(0..input.n);
        let pos2 = Coord::new(i2, j2);
        if pos1 == pos2 {
            continue;
        }
        let diff = state.calc_diff_cost(pos1, pos2, input);
        let temp = T0 + (T1 - T0) * time_keeper.get_time() / time_limit;
        if diff <= 0 || rng.gen_bool((-diff as f64 / temp).exp()) {
            state.swap(pos1, pos2);
            cost += diff;
        }
        if cost < best_cost {
            best_state = state.clone();
            best_cost = cost;
        }
    }
    let score = calc_score(input.cost, best_cost);
    eprintln!("Ideal score: {}", score);
    #[cfg(feature = "local")]
    visualizer::vis(input.n, &input.vs, &input.hs, &best_state.board.to_2d_vec());
    best_state.board
}

fn calc_score(init_cost: i64, cost: i64) -> i64 {
    let score = (init_cost as f64).log2() - (cost as f64).log2();
    let mut score = (1e6 * score).round() as i64;
    score = score.max(1);
    score
}

#[derive(Debug, Clone, Default)]
struct State {
    n: usize,
    n2: usize,
    init_pos1: Coord,
    init_pos2: Coord,
    pos1: Coord,
    pos2: Coord,
    board: DynamicMap2d<i64>,
    best_board: DynamicMap2d<i64>,
    actions: Vec<usize>,
}

impl State {
    fn new(input: &Input, pos1: Coord, pos2: Coord, best_board: DynamicMap2d<i64>) -> State {
        State {
            n: input.n,
            n2: input.n2,
            init_pos1: pos1,
            init_pos2: pos2,
            pos1,
            pos2,
            board: input.board.clone(),
            best_board,
            actions: vec![0],
        }
    }
    fn swap(&mut self, pos1: Coord, pos2: Coord) {
        let tmp = self.board[pos1];
        self.board[pos1] = self.board[pos2];
        self.board[pos2] = tmp;
    }
    fn search_different(&self, left: usize, right: usize) -> (usize, DynamicMap2d<u8>) {
        let mut different = DynamicMap2d::new_with(0, self.n);
        let mid = (left + right) / 2;
        let mut cnt = 0;
        for i in 0..self.n {
            for j in 0..self.n {
                let coord = Coord::new(i, j);
                let best_val = self.best_board[coord] as usize - 1;
                let now_val = self.board[coord] as usize - 1;
                if left <= best_val && best_val < mid && mid <= now_val && now_val < right {
                    different[coord] = 1;
                    cnt += 1;
                }
                if left <= now_val && now_val < mid && mid <= best_val && best_val < right {
                    different[coord] = 2;
                    cnt += 1;
                }
            }
        }
        cnt /= 2;
        (cnt, different)
    }
    fn quick_sort(&mut self, input: &Input) {
        let mut Q = VecDeque::new();
        Q.push_back((0, self.n2));
        while let Some((left, right)) = Q.pop_front() {
            if left == right {
                continue;
            }
            self.tsp(input, left, right);
            if self.actions.len() > 3 * 4 * self.n2 {
                break;
            }
            if right - left > 1 {
                let mid = (left + right) / 2;
                Q.push_back((left, mid));
                Q.push_back((mid, right));
            }
        }
    }
    fn tsp(&mut self, input: &Input, left: usize, right: usize) {
        let (cnt, mut different) = self.search_different(left, right);
        for _ in 0..cnt {
            let mut actions1 = self.bfs(input, self.pos1, 1, &mut different);
            let mut actions2 = self.bfs(input, self.pos2, 2, &mut different);
            assert!(!actions1.is_empty() && !actions2.is_empty());
            while actions1.len() < actions2.len() {
                actions1.push(DIRS_MAP[&'.']);
            }
            while actions1.len() > actions2.len() {
                actions2.push(DIRS_MAP[&'.']);
            }
            assert!(actions1.len() == actions2.len());
            for i in 0..actions1.len() {
                self.actions.push(actions1[i]);
                self.pos1 = self.pos1 + DIJ_DIFF[actions1[i]];
                self.actions.push(actions2[i]);
                self.pos2 = self.pos2 + DIJ_DIFF[actions2[i]];
                self.actions.push(0);
            }
            *self.actions.last_mut().unwrap() = 1;
            self.swap(self.pos1, self.pos2);
        }
        assert!(different.to_2d_vec().iter().flatten().all(|&b| b == 0));
    }
    fn bfs(
        &self,
        input: &Input,
        st: Coord,
        search: u8,
        different: &mut DynamicMap2d<u8>,
    ) -> Vec<usize> {
        let mut dist: FxHashMap<Coord, usize> = FxHashMap::default();
        dist.insert(st, 0);
        let mut Q = VecDeque::new();
        Q.push_back((st, vec![]));
        while let Some((pos, actions)) = Q.pop_front() {
            for &(dir, nxt) in input.legal_actions[pos].iter() {
                if !dist.contains_key(&nxt) || dist[&pos] + 1 < dist[&nxt] {
                    let mut nxt_actions = actions.clone();
                    nxt_actions.push(dir);
                    if different[nxt] == search {
                        different[nxt] = 0;
                        return nxt_actions;
                    }
                    dist.insert(nxt, dist[&pos] + 1);
                    Q.push_back((nxt, nxt_actions));
                }
            }
        }
        vec![]
    }
    #[fastout]
    fn output(&mut self) {
        self.actions.push(DIRS_MAP[&'.']);
        self.actions.push(DIRS_MAP[&'.']);
        self.actions.truncate(3 * 4 * self.n2);
        println!(
            "{} {} {} {}",
            self.init_pos1.row, self.init_pos1.col, self.init_pos2.row, self.init_pos2.col
        );
        // ターン毎にスワップ、高橋君移動、青木移動のアクションを同時に保存できないので、
        // タプルは使用せずに、1次元配列に保存した。
        // mod3=0: スワップ
        // mod3=1: 高橋君移動
        // mod3=2: 青木君移動
        let mut i = 0;
        while i < self.actions.len() {
            let is_swap = self.actions[i];
            let dir1 = self.actions[i + 1];
            let dir2 = self.actions[i + 2];
            println!("{} {} {}", is_swap, DIRS[dir1], DIRS[dir2]);
            i += 3;
        }
    }
}

#[derive(Debug, Clone, Default)]
struct AnnealingState {
    board: DynamicMap2d<i64>,
}

impl AnnealingState {
    fn new(board: DynamicMap2d<i64>) -> AnnealingState {
        AnnealingState { board }
    }
    fn swap(&mut self, pos1: Coord, pos2: Coord) {
        let tmp = self.board[pos1];
        self.board[pos1] = self.board[pos2];
        self.board[pos2] = tmp;
    }
    fn calc_diff_cost(&self, pos1: Coord, pos2: Coord, input: &Input) -> i64 {
        // スワップする周辺だけ差分更新
        let mut before = 0;
        for nxt in input.legal_actions[pos1].iter() {
            before += (self.board[pos1] - self.board[nxt.1]).pow(2);
        }
        for nxt in input.legal_actions[pos2].iter() {
            before += (self.board[pos2] - self.board[nxt.1]).pow(2);
        }
        let mut after = 0;
        for nxt in input.legal_actions[pos1].iter() {
            if nxt.1 == pos2 {
                after += (self.board[pos2] - self.board[pos1]).pow(2);
            } else {
                after += (self.board[pos2] - self.board[nxt.1]).pow(2);
            }
        }
        for nxt in input.legal_actions[pos2].iter() {
            if nxt.1 == pos1 {
                after += (self.board[pos1] - self.board[pos2]).pow(2);
            } else {
                after += (self.board[pos1] - self.board[nxt.1]).pow(2);
            }
        }
        after - before
    }
}

const DIRS: [char; 5] = ['U', 'D', 'L', 'R', '.'];
const DIJ: [(usize, usize); 5] = [(!0, 0), (1, 0), (0, !0), (0, 1), (0, 0)];
const DIJ_DIFF: [CoordDiff; 5] = [
    CoordDiff::new(!0, 0),
    CoordDiff::new(1, 0),
    CoordDiff::new(0, !0),
    CoordDiff::new(0, 1),
    CoordDiff::new(0, 0),
];
const DIRS_REVERSE: [usize; 5] = [1, 0, 3, 2, 4];

lazy_static::lazy_static! {
    static ref DIRS_MAP: HashMap<char, usize> = {
        let mut mp = HashMap::new();
        for (i,dir) in DIRS.iter().enumerate() {
            mp.insert(*dir, i);
        }
        mp
    };
}

struct Input {
    t: usize,
    n: usize,
    n2: usize,
    cost: i64,
    board: DynamicMap2d<i64>,
    legal_actions: DynamicMap2d<Vec<(usize, Coord)>>,
    vs: Vec<Vec<char>>,
    hs: Vec<Vec<char>>,
}

fn read_input() -> Input {
    input! {
        t: usize,
        n: usize,
        vs: [Chars; n],
        hs: [Chars; n - 1],
        board2: [[i64; n]; n]
    }

    let mut board = DynamicMap2d::new_with(0, n);
    let mut legal_actions = DynamicMap2d::new_with(vec![], n);
    let mut cost = 0;

    for i in 0..n {
        for j in 0..n {
            let coord = Coord::new(i, j);
            let coord_down = Coord::new(i + 1, j);
            let coord_right = Coord::new(i, j + 1);

            board[coord] = board2[i][j];

            if i + 1 < n && hs[i][j] == '0' {
                legal_actions[coord].push((DIRS_MAP[&'D'], coord_down));
                legal_actions[coord_down].push((DIRS_MAP[&'U'], coord));
                cost += (board2[i][j] - board2[i + 1][j]).pow(2);
            }
            if j + 1 < n && vs[i][j] == '0' {
                legal_actions[coord].push((DIRS_MAP[&'R'], coord_right));
                legal_actions[coord_right].push((DIRS_MAP[&'L'], coord));
                cost += (board2[i][j] - board2[i][j + 1]).pow(2);
            }
        }
    }

    Input {
        t,
        n,
        n2: n * n,
        cost,
        board,
        legal_actions,
        vs,
        hs,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
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

#[derive(Debug, Clone, Copy, PartialEq)]
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

#[derive(Debug, Clone, Default)]
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

impl<T: Clone> DynamicMap2d<T> {
    pub fn new_with(v: T, size: usize) -> Self {
        let map = vec![v; size * size];
        Self::new(map, size)
    }
    pub fn to_2d_vec(&self) -> Vec<Vec<T>> {
        let mut ret = vec![vec![]; self.size];
        for i in 0..self.map.len() {
            let row = i / self.size;
            ret[row].push(self.map[i].clone());
        }
        ret
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

#[cfg(feature = "local")]
mod visualizer {
    use svg::node::element::path::Data;
    use svg::node::element::{Circle, Group, Line, Path, Rectangle, Style, Text, Title};
    use svg::node::Text as TextContent;
    use svg::Document;

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

    pub fn cir(x: usize, y: usize, r: usize, fill: &str) -> Circle {
        Circle::new()
            .set("cx", x)
            .set("cy", y)
            .set("r", r)
            .set("fill", fill)
    }

    pub fn lin(x1: usize, y1: usize, x2: usize, y2: usize, color: &str) -> Line {
        Line::new()
            .set("x1", x1)
            .set("y1", y1)
            .set("x2", x2)
            .set("y2", y2)
            .set("stroke", color)
            .set("stroke-width", 3)
            .set("stroke-linecap", "round")
            .set("stroke-linecap", "round")
            .set("marker-end", "url(#arrowhead)")
        // .set("stroke-dasharray", 5)
    }

    pub fn arrow(doc: Document) -> Document {
        doc.add(TextContent::new(
        r#"<defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="3" refY="2" orient="auto">
                <polygon points="0 0, 4 2, 0 4" fill="lightgray"/>
            </marker>
        </defs>"#,
    ))
    }

    pub fn txt(x: f32, y: f32, text: &str) -> Text {
        Text::new()
            .add(TextContent::new(text))
            .set("x", x)
            .set("y", y)
            .set("fill", "black")
    }

    // 0 <= val <= 1
    pub fn color(mut val: f64) -> String {
        val = val.min(1.0);
        val = val.max(0.0);
        let (r, g, b) = if val < 0.5 {
            let x = val * 2.0;
            (
                30. * (1.0 - x) + 144. * x,
                144. * (1.0 - x) + 255. * x,
                255. * (1.0 - x) + 30. * x,
            )
        } else {
            let x = val * 2.0 - 1.0;
            (
                144. * (1.0 - x) + 255. * x,
                255. * (1.0 - x) + 30. * x,
                30. * (1.0 - x) + 70. * x,
            )
        };
        format!(
            "#{:02x}{:02x}{:02x}",
            r.round() as i32,
            g.round() as i32,
            b.round() as i32
        )
    }

    pub fn group(title: String) -> Group {
        Group::new().add(Title::new().add(TextContent::new(title)))
    }

    pub fn partition(mut doc: Document, h: &[Vec<char>], v: &[Vec<char>], size: f32) -> Document {
        let H = v.len();
        let W = h[0].len();
        for i in 0..H + 1 {
            for j in 0..W {
                // Entrance
                // if i == 0 && j == ENTRANCE {
                //     continue;
                // }
                if (i == 0 || i == H) || h[i - 1][j] == '1' {
                    let data = Data::new()
                        .move_to((size * j as f32, size * i as f32))
                        .line_by((size * 1.0, 0));
                    let p = Path::new()
                        .set("d", data)
                        .set("stroke", "black")
                        .set("stroke-width", 3.0)
                        .set("stroke-linecap", "round");
                    doc = doc.add(p);
                }
            }
        }
        for j in 0..W + 1 {
            for i in 0..H {
                // Entrance
                // if j == 0 && i == ENTRANCE {
                //     continue;
                // }
                if (j == 0 || j == W) || v[i][j - 1] == '1' {
                    let data = Data::new()
                        .move_to((size * j as f32, size * i as f32))
                        .line_by((0, size * 1.0));
                    let p = Path::new()
                        .set("d", data)
                        .set("stroke", "black")
                        .set("stroke-width", 3.0)
                        .set("stroke-linecap", "round");
                    doc = doc.add(p);
                }
            }
        }
        doc
    }

    pub fn vis(N: usize, vs: &[Vec<char>], hs: &[Vec<char>], board: &[Vec<i64>]) {
        let height = 800.0;
        let width = 800.0;
        let N2 = N * N;
        let d = height / N as f32;
        let mut doc = doc(height, width);
        doc = doc.add(Style::new(format!(
            "text {{text-anchor: middle; dominant-baseline: central; font-size: {}}}",
            10
        )));

        for i in 0..N {
            for j in 0..N {
                let rec = rect(
                    j as f32 * d,
                    i as f32 * d,
                    d,
                    d,
                    &color(board[i][j] as f64 / N2 as f64),
                );
                let text = txt(
                    j as f32 * d + d / 2.0,
                    i as f32 * d + d / 2.0,
                    &board[i][j].to_string(),
                );
                let mut grp = group(format!("(i, j) = ({}, {})\n{}", i, j, board[i][j]));
                grp = grp.add(rec);
                if N <= 20 {
                    grp = grp.add(text);
                }
                doc = doc.add(grp);
            }
        }
        doc = partition(doc, hs, vs, d);
        let vis = format!("<html><body>{}</body></html>", doc);
        std::fs::write("vis.html", vis).unwrap();
    }
}
