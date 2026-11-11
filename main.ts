// werewolf-agent-01 entry: a scripted Werewolf player (no model calls — a legal entry for this
// competition, whose harness marks model calls optional). Pure Node 22 standard library.
//
// Design for the replay rule: the agent keeps NO state between decisions. Every answer is recomputed
// from the decision's own view (the env serves the full public log and the seat's private record in
// every view), the trial seed and the game number. No clock, no Math.random, one request at a time,
// in the order the env lists them — so the live pass and the replay send identical requests.
//
// Strategy in one paragraph: parse the public record (revealed roles, speeches, role claims and the
// checks claimants report, every day's votes) into a suspicion table, then play each role's textbook
// strong line — the seer verifies claimants and leads the village once it has a wolf, the doctor
// guards the believed seer, villagers converge on the most suspicious seat instead of splitting the
// vote, and the wolves agree their kill through the first wolf's pick, hunt the village's power
// roles, bus a doomed partner late, and counterclaim seer when the real seer names one of them.

type Seat = number;
type Role = 'werewolf' | 'seer' | 'doctor' | 'villager';
type DecisionKind = 'speak' | 'vote' | 'kill' | 'protect' | 'inspect';

interface PlayerInfo {
  seat: Seat;
  name: string;
  alive: boolean;
  role: Role | null; // revealed role, once known by death or exile
}

interface Speech {
  seat: Seat;
  day: number;
  round: number;
  text: string;
}

interface ClaimedCheck {
  claimant: Seat;
  target: Seat;
  werewolf: boolean;
  day: number;
}

interface View {
  day: number;
  phase: string;
  you: {
    seat: Seat;
    name: string;
    role: Role;
    alive: boolean;
    partner: { seat: Seat; name: string; alive: boolean } | null;
  };
  players: PlayerInfo[];
  seerResults: { night: number; seat: Seat; werewolf: boolean }[] | null;
  lastProtect: { night: number; seat: Seat } | null;
  wolfVotes: { night: number; seat: Seat; target: Seat | null }[] | null;
  log: LogEntry[];
  outcome: unknown;
}

interface LogEntry {
  seq: number;
  day: number;
  kind: string;
  seat?: Seat;
  target?: Seat | null;
  round?: number;
  text?: string;
  died?: Seat | null;
  role?: Role;
  tally?: { seat: Seat; votes: number }[];
  [k: string]: unknown;
}

interface Pending {
  game: number;
  decision: DecisionKind;
  options?: Seat[];
  max_chars?: number;
  view: View;
}

/** mulberry32: a tiny deterministic PRNG. The only randomness in the entry, seeded per game. */
function rngFor(seedText: string, game: number): () => number {
  let h = 2166136261 >>> 0;
  const s = `${seedText}:${game}`;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  let a = h >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const pick = <T,>(rng: () => number, arr: readonly T[]): T => arr[Math.floor(rng() * arr.length)]!;

/** Everything the agent knows about one game, rebuilt from the view alone. */
interface GameInfo {
  view: View;
  me: Seat;
  myRole: Role;
  partner: Seat | null;
  alive: Seat[];
  revealed: Map<Seat, Role>;
  speeches: Speech[];
  passesBy: Map<Seat, number>;
  votesByDay: Map<number, { seat: Seat; target: Seat | null }[]>;
  seerClaimants: Map<Seat, ClaimedCheck[]>;
  doctorClaimants: Set<Seat>;
  checks: ClaimedCheck[]; // every check any claimed seer has reported
  accusations: Map<Seat, number>; // seat -> times named a wolf/suspect in speech (non-claim)
  defences: Map<Seat, number>; // seat -> times spoken in defence of an accused seat
}

const SEER_CLAIM = /\bi(?:'m| am) (?:the |your )?seer\b|\bas (?:the|your) seer\b|\bseer here\b|\bi am seer\b/;
const DOCTOR_CLAIM = /\bi(?:'m| am) (?:the |your )?doctor\b|\bas (?:the|your) doctor\b|\bdoctor here\b|\bi am doctor\b/;

/** The checks a speech reports, as (target, werewolf) pairs; only called on claimed seers' speeches. */
function checksIn(text: string): { target: Seat; werewolf: boolean }[] {
  const out: { target: Seat; werewolf: boolean }[] = [];
  const t = ` ${text.toLowerCase()} `;
  const patterns: [RegExp, boolean][] = [
    [/(?:seat\s*)?(\d{1,2})\s*(?:is|'s|was)\s*(?:a\s+)?werewolf\b/g, true],
    [/(?:seat\s*)?(\d{1,2})\s*(?:is|'s|was)\s*(?:a\s+)?wolf\b/g, true],
    [/checked\s*(?:seat\s*)?(\d{1,2})[^.]{0,24}?\b(?:werewolf|wolf)\b/g, true],
    [/inspect(?:ed)?\s*(?:seat\s*)?(\d{1,2})[^.]{0,24}?\b(?:werewolf|wolf)\b/g, true],
    [/(?:seat\s*)?(\d{1,2})\s*(?:is|was)\s*not\s*(?:a\s+)?(?:werewolf|wolf)\b/g, false],
    [/(?:seat\s*)?(\d{1,2})\s*(?:is|'s|was)\s*(?:a\s+)?(?:villager|innocent|human|town)\b/g, false],
    [/checked\s*(?:seat\s*)?(\d{1,2})[^.]{0,24}?\b(?:not a werewolf|innocent|villager|clear)\b/g, false],
    [/cleared\s*(?:seat\s*)?(\d{1,2})\b/g, false],
    [/(?:seat\s*)?(\d{1,2})\s*came back\s*(?:as\s*)?(?:a\s+)?(?:villager|innocent|not)/g, false],
  ];
  for (const [re, wolf] of patterns) {
    for (const m of t.matchAll(re)) {
      const seat = Number(m[1]);
      if (seat >= 1 && seat <= 8 && !out.some((c) => c.target === seat)) out.push({ target: seat, werewolf: wolf });
    }
  }
  return out;
}

/** Loose accusation scan: "seat N" near wolf-ish words in an ordinary speech. */
function accusationsIn(text: string): Seat[] {
  const out: Seat[] = [];
  const t = text.toLowerCase();
  if (!/(wolf|wolves|suspicious|suspect|lying|liar|fake|exile|vote)/.test(t)) return out;
  for (const m of t.matchAll(/seat\s*(\d{1,2})\b/g)) {
    const seat = Number(m[1]);
    if (seat >= 1 && seat <= 8 && !out.includes(seat)) out.push(seat);
  }
  return out;
}

function analyse(view: View): GameInfo {
  const revealed = new Map<Seat, Role>();
  for (const p of view.players) if (p.role) revealed.set(p.seat, p.role);
  const speeches: Speech[] = [];
  const passesBy = new Map<Seat, number>();
  const votesByDay = new Map<number, { seat: Seat; target: Seat | null }[]>();
  for (const e of view.log) {
    if (e.kind === 'speech' && typeof e.seat === 'number' && typeof e.text === 'string') {
      speeches.push({ seat: e.seat, day: e.day, round: e.round ?? 1, text: e.text });
    } else if (e.kind === 'pass' && typeof e.seat === 'number') {
      passesBy.set(e.seat, (passesBy.get(e.seat) ?? 0) + 1);
    } else if (e.kind === 'vote' && typeof e.seat === 'number') {
      const day = votesByDay.get(e.day) ?? [];
      day.push({ seat: e.seat, target: typeof e.target === 'number' ? e.target : null });
      votesByDay.set(e.day, day);
    }
  }
  // Role claims: a claimant is whoever has ever claimed; their reported checks accumulate.
  const seerClaimants = new Map<Seat, ClaimedCheck[]>();
  const doctorClaimants = new Set<Seat>();
  for (const s of speeches) {
    const low = s.text.toLowerCase();
    if (SEER_CLAIM.test(low) && !seerClaimants.has(s.seat)) seerClaimants.set(s.seat, []);
    if (DOCTOR_CLAIM.test(low)) doctorClaimants.add(s.seat);
  }
  const checks: ClaimedCheck[] = [];
  for (const s of speeches) {
    if (!seerClaimants.has(s.seat)) continue;
    for (const c of checksIn(s.text)) {
      const cc = { claimant: s.seat, target: c.target, werewolf: c.werewolf, day: s.day };
      if (!checks.some((x) => x.claimant === cc.claimant && x.target === cc.target)) {
        checks.push(cc);
        seerClaimants.get(s.seat)!.push(cc);
      }
    }
  }
  const accusations = new Map<Seat, number>();
  for (const s of speeches) {
    for (const seat of accusationsIn(s.text)) {
      if (seat !== s.seat) accusations.set(seat, (accusations.get(seat) ?? 0) + 1);
    }
  }
  return {
    view,
    me: view.you.seat,
    myRole: view.you.role,
    partner: view.you.partner ? view.you.partner.seat : null,
    alive: view.players.filter((p) => p.alive).map((p) => p.seat),
    revealed,
    speeches,
    passesBy,
    votesByDay,
    seerClaimants,
    doctorClaimants,
    checks,
    accusations,
    defences: new Map(),
  };
}

/** Is a claimed check contradicted by a revealed role? */
function checkLies(info: GameInfo, c: ClaimedCheck): boolean {
  const role = info.revealed.get(c.target);
  if (!role) return false;
  return c.werewolf !== (role === 'werewolf');
}

/** A seer claimant the village can provisionally trust: alive, uncontradicted, no proven lie. */
function believableSeers(info: GameInfo): Seat[] {
  const out: Seat[] = [];
  for (const [seat, list] of info.seerClaimants) {
    if (!info.view.players.find((p) => p.seat === seat)?.alive) continue;
    if (list.some((c) => checkLies(info, c))) continue;
    out.push(seat);
  }
  // If I am the seer, every other claimant is a liar by construction.
  if (info.myRole === 'seer') return out.filter((s) => s === info.me);
  return out;
}

/**
 * Suspicion that a living seat is a werewolf, from the village's side of the table.
 * Higher is wolfier. Ground truth (my own seer results, my partner) is applied by callers.
 */
function suspicion(info: GameInfo): Map<Seat, number> {
  const score = new Map<Seat, number>();
  const add = (seat: Seat, n: number) => score.set(seat, (score.get(seat) ?? 0) + n);
  for (const seat of info.alive) score.set(seat, 0);
  const believed = believableSeers(info);
  const soleSeer = believed.length === 1 ? believed[0]! : null;
  for (const c of info.checks) {
    if (!score.has(c.target)) continue;
    if (checkLies(info, c)) add(c.claimant, 14); // a proven liar is a wolf until shown otherwise
    else if (c.claimant === soleSeer) add(c.target, c.werewolf ? 12 : -10);
    else if (believed.includes(c.claimant)) add(c.target, c.werewolf ? 6 : -5);
    else add(c.target, c.werewolf ? 2 : -1); // an unbelieved claimant's word counts for little
  }
  for (const [day, votes] of info.votesByDay) {
    void day;
    for (const v of votes) {
      if (v.target === null || !score.has(v.seat)) continue;
      const targetRole = info.revealed.get(v.target);
      if (targetRole === 'werewolf') add(v.seat, -3);
      else if (targetRole) add(v.seat, 3); // voted to exile a known villager-side player
    }
  }
  for (const [seat, n] of info.accusations) if (score.has(seat)) add(seat, Math.min(n, 3) * 0.8);
  for (const seat of info.alive) {
    const talks = info.speeches.filter((s) => s.seat === seat).length;
    const passes = info.passesBy.get(seat) ?? 0;
    if (talks === 0 && passes >= 2) add(seat, 1.2); // the silent seats hide wolves oftener than not
    if (info.seerClaimants.has(seat) && !believed.includes(seat) && info.myRole === 'seer') add(seat, 12);
  }
  // A claimant whose checks were never contradicted and who named a revealed wolf is trusted.
  for (const seat of believed) if (score.has(seat)) add(seat, -6);
  return score;
}

/** The seats a claimant's story clears or condemns, for quick lookups. */
function claimedStatus(info: GameInfo): { cleared: Set<Seat>; named: Set<Seat> } {
  const cleared = new Set<Seat>();
  const named = new Set<Seat>();
  const believed = believableSeers(info);
  for (const c of info.checks) {
    if (!believed.includes(c.claimant) || checkLies(info, c)) continue;
    (c.werewolf ? named : cleared).add(c.target);
  }
  return { cleared, named };
}

// ---------- night decisions ----------

function decideKill(info: GameInfo, options: Seat[], rng: () => number): Seat | null {
  if (options.length === 0) return null;
  // The second wolf matches the first wolf's pick: agreement is what kills.
  const tonight = (info.view.wolfVotes ?? []).filter((v) => v.night === info.view.day);
  const partnerPick = tonight.find((v) => v.seat === info.partner && v.target !== null);
  if (partnerPick) return partnerPick.target;
  const believed = believableSeers(info);
  const { cleared } = claimedStatus(info);
  const villageTable = suspicion(info);
  const score = new Map<Seat, number>();
  for (const seat of options) {
    let s = 0;
    if (info.doctorClaimants.has(seat)) s += 9; // a claimed doctor dies before the seer they guard
    if (believed.includes(seat)) s += 8; // the believed seer
    if (info.seerClaimants.has(seat)) s += 4; // any other seer claimant
    if (cleared.has(seat)) s += 3; // a seat the village now trusts
    const talks = info.speeches.filter((x) => x.seat === seat).length;
    s += Math.min(talks, 6) * 0.4; // an influential voice
    // A seat that has accused me or my partner by name is a threat worth removing.
    const theirWords = info.speeches.filter((x) => x.seat === seat).map((x) => x.text).join(' ');
    const accusedUs = accusationsIn(theirWords);
    if (accusedUs.includes(info.me) || (info.partner !== null && accusedUs.includes(info.partner))) s += 4;
    s -= Math.max(0, villageTable.get(seat) ?? 0) * 0.5; // let the village exile its own suspects
    s += rng() * 0.7;
    score.set(seat, s);
  }
  return [...score.entries()].sort((a, b) => b[1] - a[1])[0]![0];
}

function decideProtect(info: GameInfo, options: Seat[], rng: () => number): Seat | null {
  if (options.length === 0) return null;
  const believed = believableSeers(info);
  const { cleared } = claimedStatus(info);
  const table = suspicion(info);
  const score = new Map<Seat, number>();
  for (const seat of options) {
    let s = rng() * 0.5;
    if (believed.includes(seat) && seat !== info.me) s += 10; // guard the seer the village believes
    if (cleared.has(seat)) s += 4;
    s -= (table.get(seat) ?? 0) * 0.4; // do not spend the night on the village's suspect
    s += Math.min(info.speeches.filter((x) => x.seat === seat).length, 5) * 0.2;
    score.set(seat, s);
  }
  return [...score.entries()].sort((a, b) => b[1] - a[1])[0]![0];
}

function decideInspect(info: GameInfo, options: Seat[], rng: () => number): Seat | null {
  if (options.length === 0) return null;
  const seen = new Set((info.view.seerResults ?? []).map((r) => r.seat));
  const believed = believableSeers(info);
  const { cleared, named } = claimedStatus(info);
  const table = suspicion(info);
  const score = new Map<Seat, number>();
  for (const seat of options) {
    let s = rng() * 0.6 + (table.get(seat) ?? 0) * 0.7;
    if (info.seerClaimants.has(seat) && seat !== info.me) s += 10; // verify a rival claimant first
    if (seen.has(seat)) s -= 20; // already known
    if (named.has(seat) || cleared.has(seat)) s -= 5; // the table is already acting on a claim about them
    if (believed.includes(seat)) s -= 8;
    score.set(seat, s);
  }
  return [...score.entries()].sort((a, b) => b[1] - a[1])[0]![0];
}

// ---------- day vote ----------

/** Votes already cast today, as a tally (the vote is public as it is cast). */
function todaysTally(info: GameInfo): Map<Seat, number> {
  const tally = new Map<Seat, number>();
  for (const v of info.votesByDay.get(info.view.day) ?? []) {
    if (v.target !== null) tally.set(v.target, (tally.get(v.target) ?? 0) + 1);
  }
  return tally;
}

function decideVote(info: GameInfo, options: Seat[], rng: () => number): Seat | null {
  if (options.length === 0) return null;
  const table = suspicion(info);
  const tally = todaysTally(info);
  const { cleared } = claimedStatus(info);
  const score = new Map<Seat, number>();
  if (info.myRole === 'werewolf') {
    for (const seat of options) {
      let s = (table.get(seat) ?? 0) + (tally.get(seat) ?? 0) * 2 + rng() * 0.4;
      if (seat === info.partner) {
        // Bus a partner only when the table has clearly turned and the game is not young.
        const doomed = (tally.get(seat) ?? 0) >= 3 || (table.get(seat) ?? 0) >= 10;
        s = doomed && info.view.day >= 2 ? s : -50;
      }
      score.set(seat, s);
    }
  } else {
    const myChecks = new Map((info.view.seerResults ?? []).map((r) => [r.seat, r.werewolf]));
    for (const seat of options) {
      let s = (table.get(seat) ?? 0) + (tally.get(seat) ?? 0) * 2 + rng() * 0.4;
      if (myChecks.get(seat) === true) s += 100; // I know this one myself
      if (myChecks.get(seat) === false) s -= 100;
      if (cleared.has(seat) && info.myRole !== 'seer') s -= 8;
      score.set(seat, s);
    }
  }
  const best = [...score.entries()].sort((a, b) => b[1] - a[1])[0]!;
  return best[0];
}

// ---------- debate statements ----------

const nameOf = (info: GameInfo, seat: Seat): string => info.view.players.find((p) => p.seat === seat)?.name ?? `seat ${seat}`;
const ref = (info: GameInfo, seat: Seat): string => `seat ${seat} (${nameOf(info, seat)})`;

function fold(text: string, max: number): string {
  const one = text.replace(/[\t\n\v\f\r\u0085\u2028\u2029]+/g, ' ').replace(/\p{Cc}/gu, '').trim();
  const chars = [...one];
  return chars.length > max ? chars.slice(0, max).join('').trim() : one;
}

/** A reason phrase for suspecting a seat, drawn from the public record. */
function reasonFor(info: GameInfo, seat: Seat): string {
  for (const votes of info.votesByDay.values()) {
    for (const v of votes) {
      if (v.seat === seat && v.target !== null && info.revealed.get(v.target) && info.revealed.get(v.target) !== 'werewolf') {
        return `they voted to exile ${ref(info, v.target)}, who was village`;
      }
    }
  }
  if ((info.passesBy.get(seat) ?? 0) >= 2 && info.speeches.filter((s) => s.seat === seat).length === 0) {
    return 'they have sat silent while the village talks';
  }
  if (info.seerClaimants.has(seat)) return 'their seer claim does not hold together';
  return 'their votes and words do not line up with the village';
}

/** The village-side target I would vote for right now (for statements and coordination). */
function mySpeechToday(info: GameInfo): number {
  return info.speeches.filter((s) => s.seat === info.me && s.day === info.view.day).length;
}

function seerReveal(info: GameInfo): string | null {
  const results = info.view.seerResults ?? [];
  if (results.length === 0) return null;
  const knownWolf = results.find((r) => r.werewolf && info.view.players.find((p) => p.seat === r.seat)?.alive);
  const accused = (info.accusations.get(info.me) ?? 0) >= 2;
  if (!knownWolf && info.view.day < 3 && !accused) return null;
  const parts = results.map(
    (r) => `night ${r.night} I checked ${ref(info, r.seat)}: ${r.werewolf ? 'WEREWOLF' : 'not a werewolf'}`,
  );
  let text = `I am the seer. ${parts.join('. ')}.`;
  if (knownWolf) text += ` Vote ${ref(info, knownWolf.seat)} today — that is a werewolf, and the doctor should guard me tonight.`;
  else text += ' No wolf found yet, but my clear checks are real: trust those seats.';
  return text;
}

function wolfCounterclaim(info: GameInfo): string | null {
  const { named } = claimedStatus(info);
  const threatened = named.has(info.me) || (info.partner !== null && named.has(info.partner));
  if (!threatened) return null;
  if (info.speeches.some((s) => s.seat === info.me && SEER_CLAIM.test(s.text.toLowerCase()))) {
    const accuser = [...info.seerClaimants.keys()].find((s) => s !== info.me && s !== info.partner);
    return accuser !== undefined
      ? `I have already told you: I am the seer, and ${ref(info, accuser)} is the wolf I found. Do not let a fake claim save them.`
      : 'I have already told you my checks as the seer. Judge the two claims by what the dead have shown.';
  }
  const accuser = [...info.seerClaimants.keys()].find((s) => s !== info.me && s !== info.partner);
  if (accuser === undefined) return null;
  // Fabricated history: cleared checks on revealed villagers (consistent), the accuser named last night.
  const nights: string[] = [];
  const deadVillage = [...info.revealed.entries()].filter(([, r]) => r !== 'werewolf').map(([s]) => s);
  const pastNights = Math.max(1, info.view.day - 1);
  for (let n = 1; n < pastNights && n <= deadVillage.length; n++) {
    nights.push(`night ${n} I checked ${ref(info, deadVillage[n - 1]!)}: not a werewolf`);
  }
  nights.push(`night ${pastNights} I checked ${ref(info, accuser)}: WEREWOLF`);
  return `I am the seer, and I held back until now. ${nights.join('. ')}. The claim against ${named.has(info.me) ? 'me' : ref(info, info.partner!)} is the wolves' play — exile ${ref(info, accuser)}.`;
}

function decideSpeak(info: GameInfo, maxChars: number, rng: () => number): string | null {
  const day = info.view.day;
  const round2 = mySpeechToday(info) >= 1;
  if (info.myRole === 'seer') {
    const reveal = seerReveal(info);
    if (reveal) return fold(reveal, maxChars);
  }
  if (info.myRole === 'werewolf') {
    const counter = wolfCounterclaim(info);
    if (counter) return fold(counter, maxChars);
  }
  // Ordinary play: name the table's most suspicious seat and, in round 2, commit to a vote.
  const options = info.alive.filter((s) => s !== info.me);
  const table = suspicion(info);
  if (info.myRole === 'werewolf' && info.partner !== null) table.set(info.partner, -50);
  if (info.myRole === 'seer') {
    for (const r of info.view.seerResults ?? []) table.set(r.seat, r.werewolf ? 100 : -100);
  }
  const ranked = options
    .map((s) => ({ s, v: (table.get(s) ?? 0) + rng() * 0.3 }))
    .sort((a, b) => b.v - a.v);
  const top = ranked[0];
  if (!top) return null;
  const target = top.s;
  if (info.myRole === 'werewolf' && target === info.partner) return fold('I am still weighing the claims. Let us hear everyone before we settle a vote.', maxChars);
  if (!round2) {
    if (day === 1 && info.speeches.length < 3) {
      return fold(`Day one, little to go on yet. I am ${nameOf(info, info.me)}, plain village side — I want to hear claims and watch the votes. For now ${ref(info, target)} draws my eye: ${reasonFor(info, target)}.`, maxChars);
    }
    return fold(`My strongest suspicion is ${ref(info, target)}: ${reasonFor(info, target)}. If a seer has a check, today is the day it starts to matter.`, maxChars);
  }
  return fold(`I will vote ${ref(info, target)}: ${reasonFor(info, target)}. Village, converge — a split vote exiles nobody and hands the night back to the wolves.`, maxChars);
}

// ---------- the run ----------

interface Observed {
  games: number;
  finished: number;
  done: boolean;
  pending: Pending[];
}

async function post(envUrl: string, path: string, body: unknown): Promise<{ status: number; body: any }> {
  const res = await fetch(`${envUrl}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  let parsed: any = null;
  try {
    parsed = JSON.parse(text);
  } catch {
    parsed = text;
  }
  return { status: res.status, body: parsed };
}

function actFor(p: Pending, trialSeed: string): Record<string, unknown> {
  const info = analyse(p.view);
  const rng = rngFor(`${trialSeed}:${info.view.day}:${p.decision}:${info.speeches.length}`, p.game);
  const options = p.options ?? [];
  switch (p.decision) {
    case 'speak': {
      const text = decideSpeak(info, p.max_chars ?? 500, rng);
      return text === null ? { game: p.game, decision: 'speak', pass: true } : { game: p.game, decision: 'speak', text };
    }
    case 'kill':
      return { game: p.game, decision: 'kill', target: decideKill(info, options, rng) };
    case 'protect':
      return { game: p.game, decision: 'protect', target: decideProtect(info, options, rng) };
    case 'inspect':
      return { game: p.game, decision: 'inspect', target: decideInspect(info, options, rng) };
    case 'vote':
      return { game: p.game, decision: 'vote', target: decideVote(info, options, rng) };
  }
}

const fallbackFor = (p: Pending): Record<string, unknown> =>
  p.decision === 'speak' ? { game: p.game, decision: 'speak', pass: true } : { game: p.game, decision: p.decision, target: null };

export async function run(): Promise<void> {
  const envUrl = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
  if (envUrl === '') throw new Error('MAKEFUN_ENV_URL is needed');
  const trialSeed = process.env.MAKEFUN_TRIAL_SEED ?? 'trial';
  for (;;) {
    const r = await post(envUrl, '/observe', {});
    if (r.status !== 200 || typeof r.body !== 'object' || r.body === null) {
      throw new Error(`POST /observe answered ${r.status}`);
    }
    const obs = r.body as Observed;
    if (obs.done) break;
    for (const p of obs.pending) {
      let act = actFor(p, trialSeed);
      let posted = await post(envUrl, '/act', act);
      if (posted.status === 400) {
        act = fallbackFor(p);
        posted = await post(envUrl, '/act', act);
      }
      if (posted.status !== 200 && posted.status !== 409) {
        throw new Error(`POST /act answered ${posted.status} for game ${p.game} ${p.decision}`);
      }
    }
  }
}

import { pathToFileURL } from 'node:url';

const isMain = process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  run().then(
    () => {
      process.exitCode = 0;
    },
    (err: unknown) => {
      process.stderr.write(`werewolf-agent: ${err instanceof Error ? err.stack ?? err.message : String(err)}\n`);
      process.exitCode = 1;
    },
  );
}
