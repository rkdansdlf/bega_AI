#!/usr/bin/env bash
# Apply main-branch protection with the required CI gates.
#
#   scripts/ops/apply_branch_protection.sh OWNER/REPO            # dry run (prints payload)
#   scripts/ops/apply_branch_protection.sh OWNER/REPO --apply    # needs repo admin + gh auth
#
# Required checks are the job names in .github/workflows/ci.yml.
# ci.yml's pull_request trigger has no `paths:` filter, so these checks always
# report on every PR (a path-filtered required check would hang "pending").
set -euo pipefail

repo="${1:?usage: $0 OWNER/REPO [--apply]}"
mode="${2:-}"
branch="${BRANCH:-main}"

payload="$(cat <<JSON
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "Python Linting",
      "Unit Tests",
      "Security Scan",
      "Container Image Scan"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 0,
    "dismiss_stale_reviews": true
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true
}
JSON
)"

if [[ "$mode" != "--apply" ]]; then
  echo "[dry-run] PUT repos/${repo}/branches/${branch}/protection"
  echo "$payload"
  exit 0
fi

echo "$payload" | gh api -X PUT "repos/${repo}/branches/${branch}/protection" --input -
echo "applied; verify:"
gh api "repos/${repo}/branches/${branch}/protection/required_status_checks" --jq '.contexts'
