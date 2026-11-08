-- KrishiGuard backend (Supabase): farm profiles, saved chats, and pest reports for the pest radar.
-- Every table has row level security: a farmer reads and writes only their own rows.
-- Pest reports are shared only as counts per district (pest_radar), never with who sent them.

create table if not exists public.profiles (
  id uuid primary key references auth.users on delete cascade default auth.uid(),
  data jsonb not null default '{}'::jsonb check (pg_column_size(data) < 4000),  -- the website's "My farm": name, district, crops, land, tank
  updated_at timestamptz not null default now()
);

create table if not exists public.chats (
  user_id uuid not null references auth.users on delete cascade default auth.uid(),
  id text not null check (char_length(id) <= 40),
  title text check (char_length(title) <= 120),
  msgs jsonb not null default '[]'::jsonb check (pg_column_size(msgs) < 400000),
  updated_at timestamptz not null default now(),
  primary key (user_id, id)
);

create table if not exists public.pest_reports (
  id bigint generated always as identity primary key,
  user_id uuid not null references auth.users on delete cascade default auth.uid(),
  district text not null check (char_length(district) <= 40),
  crop text check (char_length(crop) <= 40),
  pest text not null check (char_length(pest) <= 40),
  source text not null default 'chat' check (source in ('chat', 'photo')),
  created_at timestamptz not null default now()
);
create index if not exists pest_reports_recent on public.pest_reports (district, created_at desc);

alter table public.profiles enable row level security;
alter table public.chats enable row level security;
alter table public.pest_reports enable row level security;

drop policy if exists "own profile" on public.profiles;
create policy "own profile" on public.profiles for all to authenticated
  using ((select auth.uid()) = id) with check ((select auth.uid()) = id);

drop policy if exists "own chats" on public.chats;
create policy "own chats" on public.chats for all to authenticated
  using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);

drop policy if exists "add own report" on public.pest_reports;
create policy "add own report" on public.pest_reports for insert to authenticated
  with check ((select auth.uid()) = user_id);
drop policy if exists "see own reports" on public.pest_reports;
create policy "see own reports" on public.pest_reports for select to authenticated
  using ((select auth.uid()) = user_id);

grant select, insert, update, delete on public.profiles, public.chats to authenticated;
grant select, insert on public.pest_reports to authenticated;

-- at most 10 reports per person per day, and one per pest, crop and district per day, so one phone cannot flood the radar
create or replace function public.limit_pest_reports() returns trigger
language plpgsql security definer set search_path = '' as $$
begin
  if (select count(*) from public.pest_reports where user_id = new.user_id and created_at > now() - interval '1 day') >= 10 then
    raise exception 'too many reports today';
  end if;
  if exists (select 1 from public.pest_reports where user_id = new.user_id and district = new.district and pest = new.pest
             and coalesce(crop, '') = coalesce(new.crop, '') and created_at > now() - interval '1 day') then
    raise exception 'already reported today';
  end if;
  return new;
end $$;
drop trigger if exists pest_reports_limit on public.pest_reports;
create trigger pest_reports_limit before insert on public.pest_reports for each row execute function public.limit_pest_reports();

-- the radar: how many different farmers reported each pest in a district in the last N days (no user ids leave the database)
create or replace function public.pest_radar(p_district text, p_days int default 14)
returns table (pest text, crop text, farmers bigint, last_seen timestamptz)
language sql stable security definer set search_path = '' as $$
  select r.pest, r.crop, count(distinct r.user_id), max(r.created_at)
  from public.pest_reports r
  where r.district = p_district and r.created_at > now() - make_interval(days => least(greatest(p_days, 1), 60))
  group by r.pest, r.crop
  order by 3 desc, 4 desc
  limit 20
$$;
revoke all on function public.pest_radar(text, int) from public;
grant execute on function public.pest_radar(text, int) to anon, authenticated;
revoke all on function public.limit_pest_reports() from public, anon, authenticated;
