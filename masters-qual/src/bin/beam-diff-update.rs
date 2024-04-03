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
    let mut input = read_input(&mut rng);

    let mut best_score = std::i64::MAX;
    let mut best_actions = vec![];
    let mut best_init_pos1 = Coord::default();
    let mut best_init_pos2 = Coord::default();

    while !time_keeper.isTimeOver() {
        let init_pos1 = init_pos(&input, &mut rng);
        let init_pos2 = init_pos(&input, &mut rng);

        let init_state = State {
            pos1: init_pos1,
            pos2: init_pos2,
            board: input.board.clone(),
        };

        let mut init_hash = 0;
        init_hash ^= input.pos1_hash[init_pos1];
        init_hash ^= input.pos2_hash[init_pos2];

        for i in 0..input.n {
            for j in 0..input.n {
                let coord = Coord::new(i, j);
                let num = input.board[coord] as usize;
                let hash = rng.gen::<u64>();
                init_hash ^= hash;
                input.hashes[num].insert(coord, hash);
            }
        }

        let MAX_TURN = 4 * input.n * input.n;
        let width = if input.t == 19 { 15 } else { 1e2 as usize };
        let depth = if input.t == 19 { 50 } else { 30 };
        let mut actions = vec![0];
        let mut action_cnt = 1;
        let mut init_node = Node {
            track_id: !0,
            refs: 0,
            score: input.cost,
            hash: init_hash,
            state: init_state,
        };

        while action_cnt < MAX_TURN && !time_keeper.isTimeOver() {
            let mut beam = BeamSearch::new(init_node, width, depth);
            let (action, best_node) =
                beam.solve(depth.min(MAX_TURN - action_cnt), &mut input, &mut rng);

            for row in action.iter() {
                actions.push(row.dir1);
                actions.push(row.dir2);
                actions.push(row.swap);
            }
            action_cnt += action.len();
            init_node = best_node;
            init_node.track_id = !0;
            init_node.refs = 0;
        }
        actions.push(DIRS_MAP[&'.']);
        actions.push(DIRS_MAP[&'.']);

        if init_node.score < best_score {
            best_score = init_node.score;
            best_actions = actions.clone();
            best_init_pos1 = init_pos1;
            best_init_pos2 = init_pos2;
        }
    }

    println!(
        "{} {} {} {}",
        best_init_pos1.row, best_init_pos1.col, best_init_pos2.row, best_init_pos2.col
    );
    let mut i = 0;
    while i < best_actions.len() {
        let is_swap = best_actions[i];
        let dir1 = best_actions[i + 1];
        let dir2 = best_actions[i + 2];
        println!("{} {} {}", is_swap, DIRS[dir1], DIRS[dir2]);
        i += 3;
    }

    let score = (input.cost as f64).log2() - (best_score as f64).log2();
    let mut score = (1e6 * score).round() as usize;
    score = score.max(1);
    eprintln!("Score: {}", score);

    #[allow(unused_mut, unused_assignments)]
    let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
    #[cfg(feature = "local")]
    {
        eprintln!("Local Mode");
        elapsed_time *= 0.55;
    }
    eprintln!("Elapsed: {}", (elapsed_time * 1000.0) as usize);
}

#[derive(Debug, Clone, Default)]
struct State {
    pos1: Coord,
    pos2: Coord,
    board: DynamicMap2d<i64>,
}

impl State {
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

#[derive(Debug, Clone, Default)]
struct Node {
    track_id: usize,
    refs: usize,
    score: i64,
    hash: u64,
    state: State,
}
impl Node {
    fn new_node(&self, cand: &Cand) -> Node {
        let mut ret = self.clone();
        ret.apply(cand);
        ret
    }
    fn apply(&mut self, cand: &Cand) {
        self.state.pos1 = cand.op.pos1;
        self.state.pos2 = cand.op.pos2;
        if cand.op.swap == 1 {
            let tmp = self.state.board[self.state.pos1];
            self.state.board[self.state.pos1] = self.state.board[self.state.pos2];
            self.state.board[self.state.pos2] = tmp;
        }
        self.score = cand.eval_score;
        self.hash = cand.hash;
    }
}

#[derive(Debug, Clone, Copy)]
struct Op {
    dir1: usize,
    pos1: Coord,
    dir2: usize,
    pos2: Coord,
    swap: usize,
}

#[derive(Debug, Clone)]
struct Cand {
    op: Op,
    parent: usize,
    eval_score: i64,
    hash: u64,
}
impl Cand {
    fn raw_score(&self, _input: &Input) -> i64 {
        self.eval_score
    }
}

#[derive(Debug)]
struct BeamSearch {
    track: Vec<(usize, Op)>,
    nodes: Vec<Node>,
    free: Vec<usize>,
    at: usize,
    cands: Vec<Cand>,
    max_width: usize,
}
impl BeamSearch {
    fn new(node: Node, max_width: usize, max_turn: usize) -> BeamSearch {
        let max_nodes = max_width * max_turn;
        let mut nodes = vec![Node::default(); max_width * 2];
        nodes[0] = node;
        BeamSearch {
            free: (0..nodes.len()).collect(),
            nodes,
            at: 1,
            track: Vec::with_capacity(max_nodes),
            cands: Vec::with_capacity(max_width),
            max_width,
        }
    }

    fn enum_cands(&self, input: &mut Input, cands: &mut Vec<Cand>, rng: &mut rand_pcg::Pcg64Mcg) {
        for &i in &self.free[..self.at] {
            self.append_cands(input, i, cands, rng);
        }
    }

    fn append_cands(
        &self,
        input: &mut Input,
        parent_idx: usize,
        cands: &mut Vec<Cand>,
        rng: &mut rand_pcg::Pcg64Mcg,
    ) {
        let parent_node = &self.nodes[parent_idx];
        let parent_score = parent_node.score;
        let parent_hash = parent_node.hash;
        let pos1 = parent_node.state.pos1;
        let pos2 = parent_node.state.pos2;
        for &(dir1, nxt1) in input.legal_actions[pos1].iter() {
            for &(dir2, nxt2) in input.legal_actions[pos2].iter() {
                if dir1 == dir2 && dir1 == DIRS_MAP[&'.'] {
                    continue;
                }
                let mut next_hash = parent_hash;
                next_hash ^= input.pos1_hash[pos1];
                next_hash ^= input.pos1_hash[nxt1];
                next_hash ^= input.pos2_hash[pos2];
                next_hash ^= input.pos2_hash[nxt2];

                for swap in 0..2 {
                    let diff_score = if swap == 0 {
                        0
                    } else {
                        next_hash ^= input.hashes[parent_node.state.board[nxt1] as usize][&nxt1];
                        #[allow(clippy::map_entry)]
                        if input.hashes[parent_node.state.board[nxt1] as usize].contains_key(&nxt2)
                        {
                            next_hash ^=
                                input.hashes[parent_node.state.board[nxt1] as usize][&nxt2];
                        } else {
                            let hash = rng.gen::<u64>();
                            next_hash ^= hash;
                            input.hashes[parent_node.state.board[nxt1] as usize].insert(nxt2, hash);
                        }

                        next_hash ^= input.hashes[parent_node.state.board[nxt2] as usize][&nxt2];
                        #[allow(clippy::map_entry)]
                        if input.hashes[parent_node.state.board[nxt2] as usize].contains_key(&nxt1)
                        {
                            next_hash ^=
                                input.hashes[parent_node.state.board[nxt2] as usize][&nxt1];
                        } else {
                            let hash = rng.gen::<u64>();
                            next_hash ^= hash;
                            input.hashes[parent_node.state.board[nxt2] as usize].insert(nxt1, hash);
                        }

                        parent_node.state.calc_diff_cost(nxt1, nxt2, input)
                    };
                    let op = Op {
                        dir1,
                        pos1: nxt1,
                        dir2,
                        pos2: nxt2,
                        swap,
                    };

                    let cand = Cand {
                        op,
                        parent: parent_idx,
                        eval_score: parent_score + diff_score,
                        hash: next_hash,
                    };
                    cands.push(cand);
                }
            }
        }
    }

    fn update<I: Iterator<Item = Cand>>(&mut self, cands: I) {
        self.cands.clear();
        for cand in cands {
            self.nodes[cand.parent].refs += 1;
            self.cands.push(cand);
        }

        for i in (0..self.at).rev() {
            if self.nodes[self.free[i]].refs == 0 {
                self.at -= 1;
                self.free.swap(i, self.at);
            }
        }

        for cand in &self.cands {
            let node = &mut self.nodes[cand.parent];
            node.refs -= 1;
            let prev = node.track_id;

            let new = if node.refs == 0 {
                node.apply(cand);
                node
            } else {
                let mut new = node.new_node(cand);
                new.refs = 0;
                let idx = self.free[self.at];
                self.at += 1;
                self.nodes[idx] = new;
                &mut self.nodes[idx]
            };

            self.track.push((prev, cand.op));
            new.track_id = self.track.len() - 1;
        }
    }

    fn restore(&self, mut idx: usize) -> Vec<Op> {
        idx = self.nodes[idx].track_id;
        let mut ret = vec![];
        while idx != !0 {
            ret.push(self.track[idx].1);
            idx = self.track[idx].0;
        }
        ret.reverse();
        ret
    }

    fn solve(
        &mut self,
        depth: usize,
        input: &mut Input,
        rng: &mut rand_pcg::Pcg64Mcg,
    ) -> (Vec<Op>, Node) {
        let mut cands = Vec::<Cand>::new();
        let mut set = rustc_hash::FxHashSet::default();
        for t in 0..depth {
            if t != 0 {
                cands.sort_unstable_by_key(|a| a.eval_score);
                set.clear();
                self.update(
                    cands
                        .iter()
                        .filter(|cand| set.insert(cand.hash))
                        .take(self.max_width)
                        .cloned(),
                );
            }
            cands.clear();
            self.enum_cands(input, &mut cands, rng);
        }

        let best = cands.iter().min_by_key(|a| a.raw_score(input)).unwrap();
        let parent_node = &self.nodes[best.parent];
        let best_node = parent_node.new_node(best);
        let mut ret = self.restore(best.parent);
        ret.push(best.op);
        (ret, best_node)
    }
}

fn init_pos(input: &Input, rng: &mut rand_pcg::Pcg64Mcg) -> Coord {
    loop {
        let i = rng.gen_range(0..input.n);
        let j = rng.gen_range(0..input.n);
        let coord = Coord::new(i, j);
        if input.legal_actions[coord].len() > 1 {
            return coord;
        }
    }
}

const DIRS: [char; 5] = ['U', 'D', 'L', 'R', '.'];
const DIJ: [(usize, usize); 5] = [(!0, 0), (1, 0), (0, !0), (0, 1), (0, 0)];
const DIRS_REVERSE: [usize; 4] = [1, 0, 3, 2];

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
    cost: i64,
    board: DynamicMap2d<i64>,
    legal_actions: DynamicMap2d<Vec<(usize, Coord)>>,
    pos1_hash: DynamicMap2d<u64>,
    pos2_hash: DynamicMap2d<u64>,
    hashes: Vec<FxHashMap<Coord, u64>>,
}

fn read_input(rng: &mut rand_pcg::Pcg64Mcg) -> Input {
    input! {
        t: usize,
        n: usize,
        v: [Chars; n],
        h: [Chars; n - 1],
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

            if i + 1 < n && h[i][j] == '0' {
                legal_actions[coord].push((DIRS_MAP[&'D'], coord_down));
                legal_actions[coord_down].push((DIRS_MAP[&'U'], coord));
                cost += (board2[i][j] - board2[i + 1][j]).pow(2);
            }
            if j + 1 < n && v[i][j] == '0' {
                legal_actions[coord].push((DIRS_MAP[&'R'], coord_right));
                legal_actions[coord_right].push((DIRS_MAP[&'L'], coord));
                cost += (board2[i][j] - board2[i][j + 1]).pow(2);
            }
        }
    }

    let mut pos1_hash = DynamicMap2d::new_with(0, n);
    let mut pos2_hash = DynamicMap2d::new_with(0, n);
    for i in 0..n {
        for j in 0..n {
            let coord = Coord::new(i, j);
            pos1_hash[coord] = rng.gen::<u64>();
            pos2_hash[coord] = rng.gen::<u64>();
        }
    }

    Input {
        t,
        n,
        cost,
        board,
        legal_actions,
        pos1_hash,
        pos2_hash,
        hashes: vec![FxHashMap::default(); n * n + 1],
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
