#!/bin/bash

# Default argument values
USER_COUNT=1000
MOVIE_COUNT=250
C_VALUE=0.045
MIN_RATING=4.0

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --user_count) USER_COUNT="$2"; shift ;;
        --movie_count) MOVIE_COUNT="$2"; shift ;;
        --c_value) C_VALUE="$2"; shift ;;
        --min_rating) MIN_RATING="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Running ising_model.py with:"
echo "  User count: $USER_COUNT"
echo "  Movie count: $MOVIE_COUNT"
echo "  C value: $C_VALUE"
echo "  Min rating: $MIN_RATING"

python3 ising_model.py --user_count "$USER_COUNT" --movie_count "$MOVIE_COUNT" --c-value "$C_VALUE" --min-rating "$MIN_RATING"