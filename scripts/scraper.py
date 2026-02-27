import asyncio
import aiohttp
import csv
import json
import os
from typing import Any

URL = "https://graphql-gateway.production-eks.codecademy.com/"

HEADERS = {
    "accept": "*/*",
    "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
    "apollographql-client-name": "portal-app",
    "apollographql-client-version": "c062790d7e9844ede542584827da85c308c73f02",
    "content-type": "application/json",
    "origin": "https://www.codecademy.com",
    "referer": "https://www.codecademy.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
}

# Paste your CSRF-TOKEN and cookie here if you want enrollmentStatus data.
# Leave empty strings to skip enrollment data (works without auth).
COOKIE = ""
CSRF_TOKEN = ""

QUERY = """
query paginatedCatalog(
  $difficulty: [ContainerDifficultyEnum!],
  $proExclusive: Boolean,
  $consumerCatalogContainerTypes: [ConsumerCatalogContainer!],
  $minDurationHours: Int,
  $maxDurationHours: Int,
  $order: ContainerOrderBy,
  $paginate: PaginationInput!,
  $skipEnrollmentStatus: Boolean = false
) {
  paginatedContainers(
    difficulty: $difficulty
    proExclusive: $proExclusive
    consumerCatalogContainerTypes: $consumerCatalogContainerTypes
    minDurationHours: $minDurationHours
    maxDurationHours: $maxDurationHours
    order: $order
    paginate: $paginate
  ) {
    collection {
      ... on Track {
        ...rawTrackCardData
        __typename
      }
      ... on Path {
        ...rawPathCardData
        __typename
      }
      ... on ExternalCourse {
        ...rawExtCourseCardData
        __typename
      }
      ... on ExternalPath {
        ...rawExtPathCardData
        __typename
      }
      __typename
    }
    paginationMetadata {
      totalPages
      totalResults
      __typename
    }
    __typename
  }
}

fragment rawTrackCardData on Track {
  id
  slug
  title
  lessonCount: contentItemCount(contentItemType: [Lesson])
  contentModuleIds
  grantsCertificate
  enrollmentStatus @skip(if: $skipEnrollmentStatus)
  pro: proExclusive
  shortDescription
  difficulty
  metrics {
    medianDurationHours
    __typename
  }
  __typename
}

fragment rawPathCardData on Path {
  id
  slug
  goal
  title
  lessonCount: contentItemCount(contentItemType: [Lesson])
  trackCount
  enrollmentStatus @skip(if: $skipEnrollmentStatus)
  shortDescription
  difficulty
  metrics {
    medianDurationHours
    __typename
  }
  __typename
}

fragment rawExtCourseCardData on ExternalCourse {
  id
  slug
  title
  grantsCertificate
  enrollmentStatus @skip(if: $skipEnrollmentStatus)
  pro: proExclusive
  difficulty
  durationHours
  shortDescription
  longDescription
  __typename
}

fragment rawExtPathCardData on ExternalPath {
  id
  slug
  title
  classification
  courseCount
  enrollmentStatus @skip(if: $skipEnrollmentStatus)
  difficulty
  durationHours
  shortDescription
  longDescription
  certificationProvider
  __typename
}
"""

CONTAINER_TYPES = [
    "CAREER_PATH",
    "EXTERNAL_CERTIFICATION_PATH",
    "EXTERNAL_COURSE",
    "EXTERNAL_JOURNEY_PATH",
    "SKILL_PATH",
    "TRACK",
]

PER_PAGE = 48

CSV_FIELDS = [
    "id", "type", "slug", "title", "shortDescription", "longDescription",
    "difficulty", "pro", "grantsCertificate", "enrollmentStatus",
    "lessonCount", "trackCount", "courseCount",
    "medianDurationHours", "durationHours",
    "goal", "classification", "certificationProvider",
    "contentModuleIds",
]


def flatten_item(item: dict[str, Any]) -> dict[str, Any]:
    typename = item.get("__typename", "")
    metrics = item.get("metrics") or {}
    module_ids = item.get("contentModuleIds")
    return {
        "id": item.get("id", ""),
        "type": typename,
        "slug": item.get("slug", ""),
        "title": item.get("title", ""),
        "shortDescription": item.get("shortDescription", ""),
        "longDescription": item.get("longDescription", ""),
        "difficulty": item.get("difficulty", ""),
        "pro": item.get("pro", ""),
        "grantsCertificate": item.get("grantsCertificate", ""),
        "enrollmentStatus": item.get("enrollmentStatus", ""),
        "lessonCount": item.get("lessonCount", ""),
        "trackCount": item.get("trackCount", ""),
        "courseCount": item.get("courseCount", ""),
        "medianDurationHours": metrics.get("medianDurationHours", ""),
        "durationHours": item.get("durationHours", ""),
        "goal": item.get("goal", ""),
        "classification": item.get("classification", ""),
        "certificationProvider": item.get("certificationProvider", ""),
        "contentModuleIds": json.dumps(module_ids) if module_ids else "",
    }


def build_payload(page: int, skip_enrollment: bool) -> dict:
    return {
        "operationName": "paginatedCatalog",
        "query": QUERY,
        "variables": {
            "skipEnrollmentStatus": skip_enrollment,
            "consumerCatalogContainerTypes": CONTAINER_TYPES,
            "order": {"by": "RECENT_ENROLLMENT_COUNT", "direction": "DESC"},
            "paginate": {"perPage": PER_PAGE, "page": page},
        },
    }


async def fetch_page(
    session: aiohttp.ClientSession,
    page: int,
    skip_enrollment: bool,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        payload = build_payload(page, skip_enrollment)
        async with session.post(URL, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def scrape_all() -> list[dict]:
    skip_enrollment = not bool(COOKIE and CSRF_TOKEN)
    if skip_enrollment:
        print("No cookie/CSRF-TOKEN set — skipping enrollmentStatus field.")

    headers = dict(HEADERS)
    if COOKIE:
        headers["cookie"] = COOKIE
    if CSRF_TOKEN:
        headers["x-csrf-token"] = CSRF_TOKEN

    semaphore = asyncio.Semaphore(10)  # max 10 concurrent requests

    async with aiohttp.ClientSession(headers=headers) as session:
        # Fetch page 1 to discover total pages
        print("Fetching page 1 to get pagination metadata...")
        first = await fetch_page(session, 1, skip_enrollment, semaphore)

        data = first["data"]["paginatedContainers"]
        meta = data["paginationMetadata"]
        total_pages = meta["totalPages"]
        total_results = meta["totalResults"]
        print(f"Total results: {total_results}  |  Total pages: {total_pages}")

        all_items = list(data["collection"])

        if total_pages > 1:
            tasks = [
                fetch_page(session, page, skip_enrollment, semaphore)
                for page in range(2, total_pages + 1)
            ]
            print(f"Fetching remaining {len(tasks)} pages concurrently...")
            results = await asyncio.gather(*tasks)
            for r in results:
                all_items.extend(r["data"]["paginatedContainers"]["collection"])

    print(f"Collected {len(all_items)} items total.")
    return all_items


def save_csv(items: list[dict], output_path: str) -> None:
    rows = [flatten_item(item) for item in items]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {output_path}")


async def main() -> None:
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "codecademy.csv")

    items = await scrape_all()
    save_csv(items, output_path)


if __name__ == "__main__":
    asyncio.run(main())
