#!/usr/bin/env bash
# Shared config for all model runs
INPUT="core_tests.json"
TRIALS=3
CONCURRENT=30
RETRIES=2
RETRY_DELAY=5
TIER="core"
COMMON_ARGS="-i $INPUT -n $TRIALS -c $CONCURRENT --rate-limit-retries $RETRIES --rate-limit-retry-delay $RETRY_DELAY --tier $TIER"

mkdir -p results
