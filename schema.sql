--
-- PostgreSQL database dump
--

\restrict oj5kJGBWMpIp7FhRO1qUU0Ma25UIraEHVokuN7EXrMhEaMknch9O23KUbDo3iQd

-- Dumped from database version 17.7 (Homebrew)
-- Dumped by pg_dump version 17.7 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: issue_links; Type: TABLE; Schema: public; Owner: selmadjozic
--

CREATE TABLE public.issue_links (
    source_key text,
    direction text,
    link_type text,
    target_key text
);


ALTER TABLE public.issue_links OWNER TO selmadjozic;

--
-- Name: issues; Type: TABLE; Schema: public; Owner: selmadjozic
--

CREATE TABLE public.issues (
    summary text,
    issue_key text,
    issue_id text,
    issue_type text,
    status text,
    priority text,
    resolution text,
    reporter text,
    created timestamp without time zone,
    updated timestamp without time zone,
    resolved timestamp without time zone,
    affects_version text,
    fix_version text,
    component text,
    votes bigint,
    labels text,
    description text,
    environment text,
    severity text,
    days_to_resolve double precision,
    has_attachment boolean,
    desc_len bigint,
    time_band text
);


ALTER TABLE public.issues OWNER TO selmadjozic;

--
-- Name: label_stats; Type: TABLE; Schema: public; Owner: selmadjozic
--

CREATE TABLE public.label_stats (
    label text,
    n_labelled bigint,
    n_classes bigint,
    majority_class text,
    majority_share double precision
);


ALTER TABLE public.label_stats OWNER TO selmadjozic;

--
-- Name: issue_links_source_key_idx; Type: INDEX; Schema: public; Owner: selmadjozic
--

CREATE INDEX issue_links_source_key_idx ON public.issue_links USING btree (source_key);


--
-- Name: issue_links_target_key_idx; Type: INDEX; Schema: public; Owner: selmadjozic
--

CREATE INDEX issue_links_target_key_idx ON public.issue_links USING btree (target_key);


--
-- Name: issues_created_idx; Type: INDEX; Schema: public; Owner: selmadjozic
--

CREATE INDEX issues_created_idx ON public.issues USING btree (created);


--
-- Name: issues_issue_key_idx; Type: INDEX; Schema: public; Owner: selmadjozic
--

CREATE INDEX issues_issue_key_idx ON public.issues USING btree (issue_key);


--
-- PostgreSQL database dump complete
--

\unrestrict oj5kJGBWMpIp7FhRO1qUU0Ma25UIraEHVokuN7EXrMhEaMknch9O23KUbDo3iQd

