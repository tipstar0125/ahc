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

mod rnd {
    static mut S: usize = 0;
    static MAX: usize = 1e9 as usize;

    #[inline]
    pub fn init(seed: usize) {
        unsafe {
            if seed == 0 {
                let t = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as usize;
                S = t
            } else {
                S = seed;
            }
        }
    }
    #[inline]
    pub fn gen() -> usize {
        unsafe {
            if S == 0 {
                init(0);
            }
            S ^= S << 7;
            S ^= S >> 9;
            S
        }
    }
    #[inline]
    pub fn gen_range(a: usize, b: usize) -> usize {
        gen() % (b - a) + a
    }
    #[inline]
    pub fn gen_bool() -> bool {
        gen() & 1 == 1
    }
    #[inline]
    pub fn gen_range_isize(a: usize) -> isize {
        let mut x = (gen() % a) as isize;
        if gen_bool() {
            x *= -1;
        }
        x
    }
    #[inline]
    pub fn gen_range_neg_wrapping(a: usize) -> usize {
        let mut x = gen() % a;
        if gen_bool() {
            x = x.wrapping_neg();
        }
        x
    }
    #[inline]
    pub fn gen_float() -> f64 {
        ((gen() % MAX) as f64) / MAX as f64
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
            elapsed_time * 1.5 >= self.time_threshold
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
            elapsed_time * 1.5
        }
        #[cfg(not(feature = "local"))]
        {
            elapsed_time
        }
    }
}

const H: usize = 60;
const W: usize = 25;
const TURN: usize = 1000;
const INF: isize = 1 << 60;
const TARGET_LEVEL: isize = 250;

#[derive(Debug, Clone, Eq, PartialEq)]
struct Player {
    x: usize,
    y: usize,
}

impl Player {
    fn new() -> Self {
        Player { x: 12, y: 0 }
    }
    fn move_(&mut self, action: isize) {
        self.y += 1;
        self.x = (W as isize + action + self.x as isize) as usize % W;
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct Enemy {
    hp: isize,
    power: isize,
    init_hp: isize,
    x: usize,
    y: usize,
}

impl Enemy {
    fn new(init_hp: isize, power: isize, x: usize, y: usize) -> Self {
        Enemy {
            hp: init_hp,
            power,
            init_hp,
            x,
            y,
        }
    }
}

#[derive(Debug, Clone, Eq)]
struct State {
    player: Player,
    S: isize,
    score: isize,
    damage: isize,
    evaluated_score: isize,
    enemies: Vec<VecDeque<Enemy>>,
    turn: usize,
    first_action: isize,
    is_dead: bool,
}

impl State {
    fn new() -> Self {
        State {
            player: Player::new(),
            S: 0,
            score: 0,
            damage: 0,
            evaluated_score: 0,
            enemies: vec![VecDeque::new(); W],
            turn: 0,
            first_action: 0,
            is_dead: false,
        }
    }
    fn update_enemy(&mut self, n: usize) {
        let y = H + self.turn;
        for _ in 0..n {
            let v: Vec<isize> = read_vec();
            let h = v[0];
            let p = v[1];
            let x = v[2] as usize;
            let enemy = Enemy::new(h, p, x, y);
            self.enemies[x].push_back(enemy);
        }
        for x in 0..W {
            if self.enemies[x].is_empty() {
                continue;
            }
            if y - self.enemies[x][0].y == H {
                self.enemies[x].pop_front();
            }
        }
    }
    fn get_level(&self) -> isize {
        1 + self.S / 100
    }
    fn advance(&mut self, action: isize) {
        self.player.move_(action);
        self.attack();
        self.turn += 1;
    }
    fn attack(&mut self) {
        if self.enemies[self.player.x].is_empty() {
            return;
        }
        let level = self.get_level();
        let enemy = &self.enemies[self.player.x][0];
        if enemy.y == self.player.y {
            self.is_dead = true;
            return;
        }
        if enemy.y == self.player.y + 1 && enemy.hp > level {
            self.is_dead = true;
            return;
        }
        if enemy.hp <= level {
            self.score += enemy.init_hp;
            self.S += enemy.power;
            self.damage += min!(level, enemy.hp);
            self.enemies[self.player.x].pop_front();
        } else {
            self.enemies[self.player.x][0].hp -= level;
            self.damage += level;
        }
    }
    fn evaluate_score(&mut self) {
        let level = self.get_level();
        self.evaluated_score = self.S * 1e6 as isize + self.damage;

        if self.enemies[self.player.x].is_empty() {
            return;
        }
        let hp = self.enemies[self.player.x][0].hp;
        let enemy_y = self.enemies[self.player.x][0].y;
        let player_y = self.player.y;
        let turn = enemy_y - player_y;
        if hp <= level * turn as isize {
            self.evaluated_score += 1000;
        }
    }
    fn is_done(&self) -> bool {
        self.turn == TURN
    }
    fn output(&self, action: isize) {
        if action == 0 {
            println!("S");
        } else if action == -1 {
            println!("L");
        } else {
            println!("R");
        }
    }
}

impl std::cmp::PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.evaluated_score == other.evaluated_score
    }
}

impl std::cmp::PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.evaluated_score == other.evaluated_score {
            Some(std::cmp::Ordering::Equal)
        } else if self.evaluated_score > other.evaluated_score {
            Some(std::cmp::Ordering::Greater)
        } else if self.evaluated_score < other.evaluated_score {
            Some(std::cmp::Ordering::Less)
        } else {
            None
        }
    }
}

impl std::cmp::Ord for State {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.evaluated_score == other.evaluated_score {
            std::cmp::Ordering::Equal
        } else if self.evaluated_score > other.evaluated_score {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        }
    }
}

fn beam_search_action(state: &State, beam_width: usize, time_threshold: f64) -> isize {
    let mut now_beam = BinaryHeap::new();
    let mut best_state = state;
    now_beam.push(state.clone());
    let time_keeper = TimeKeeper::new(time_threshold);
    let mut turn = 0;

    for t in 0.. {
        let mut next_beam = BinaryHeap::new();
        turn = t + 1;

        for _ in 0..beam_width {
            if now_beam.is_empty() {
                break;
            }
            let now_state = now_beam.pop().unwrap();
            let actions = vec![0, -1, 1];
            for action in actions {
                let mut next_state = now_state.clone();
                if t == 0 {
                    next_state.first_action = action;
                    next_state.damage = 0;
                }
                next_state.advance(action);
                next_state.update_enemy(0);
                next_state.evaluate_score();
                if !next_state.is_dead {
                    next_beam.push(next_state);
                }
            }

            if time_keeper.isTimeOver() {
                break;
            }
        }
        now_beam = next_beam;
        best_state = now_beam.peek().unwrap();
        if best_state.is_done() || time_keeper.isTimeOver() {
            break;
        }
    }
    #[cfg(feature = "local")]
    {
        eprintln!("{}", turn);
    }
    best_state.first_action
}

#[derive(Default)]
struct Solver {}
impl Solver {
    fn solve(&mut self) {
        let mut state = State::new();

        #[cfg(feature = "local")]
        {
            eprintln!("Local Mode");
            let _: Vec<usize> = read_vec();
        }

        let start = std::time::Instant::now();
        let time_threshold = 1.0 * 1e-3; // [sec]

        while !state.is_done() {
            let N: isize = read();
            if N == -1 {
                return;
            }

            state.update_enemy(N as usize);
            let action = beam_search_action(&state, 10000, time_threshold);
            state.advance(action);
            state.output(action);
        }
        eprintln!("Score: {}", state.score);
        eprintln!("S: {}", state.S);

        #[allow(unused_mut, unused_assignments)]
        let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
        #[cfg(feature = "local")]
        {
            eprintln!("Local Mode");
            elapsed_time *= 1.5;
        }
        eprintln!("Elapsed time: {}sec", elapsed_time);
    }
}

#[macro_export]
macro_rules! max {
    ($x: expr) => ($x);
    ($x: expr, $( $y: expr ),+) => {
        std::cmp::max($x, max!($( $y ),+))
    }
}
#[macro_export]
macro_rules! min {
    ($x: expr) => ($x);
    ($x: expr, $( $y: expr ),+) => {
        std::cmp::min($x, min!($( $y ),+))
    }
}

fn main() {
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(|| Solver::default().solve())
        .unwrap()
        .join()
        .unwrap();
}

fn read<T: std::str::FromStr>() -> T {
    let mut s = String::new();
    std::io::stdin().read_line(&mut s).ok();
    s.trim().parse().ok().unwrap()
}

fn read_vec<T: std::str::FromStr>() -> Vec<T> {
    read::<String>()
        .split_whitespace()
        .map(|e| e.parse().ok().unwrap())
        .collect()
}
