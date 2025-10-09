#!/usr/bin/env python3
import os
import pandas as pd
import polars as pl
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan


def get_data_from_elastic(es, source_index):
    query = {
        "track_total_hits": True,
        "query": {
            "bool": {
                "filter": [
                    {
                        "match": {
                            "elastic.owner.team": "awesome-sre-team"
                        }
                    },
                    {
                        "match": {
                            "elastic.environment": "production"
                        }
                    },
                    {
                        "terms": {
                            "vulnerability.severity": [
                                "Critical",
                                "critical",
                                "High",
                                "high",
                                "Medium",
                                "medium",
                                "Low",
                                "low"
                            ]
                        }
                    }
                ],
            }
        },
        "fields": [
            "qualys_vmdr.asset_host_detection.vulnerability.unique_vuln_id",
            "qualys_vmdr.asset_host_detection.vulnerability.first_found_datetime",
            "qualys_vmdr.asset_host_detection.vulnerability.last_found_datetime",
            "qualys_vmdr.asset_host_detection.vulnerability.published_datetime",
            "vulnerability.severity"
        ],
        "script_fields": {
            "fixed": {
                "script": {
                    "source": "if (doc['qualys_vmdr.asset_host_detection.vulnerability.status'].value == 'Fixed') { return 1; } else { return 0; }",
                    "lang": "painless"
                }
            },
            "vulnerability_age": {
                "script": {
                    "source": "ZonedDateTime last_found = ZonedDateTime.parse(doc['qualys_vmdr.asset_host_detection.vulnerability.last_found_datetime'].value.toString()); ZonedDateTime first_found = ZonedDateTime.parse(doc['qualys_vmdr.asset_host_detection.vulnerability.first_found_datetime'].value.toString()); return ChronoUnit.DAYS.between(first_found, last_found);",
                    "lang": "painless"
                }
            }
        },
        "_source": False
    }

    # Use the scan helper to efficiently scroll through all results.
    results = list(scan(
        client=es,
        query=query,
        scroll='30m',
        index=source_index,
        size=10000,
        raise_on_error=True,
        preserve_order=False,
        clear_scroll=True
    ))

    data = []
    for hit in results:
        if "fields" not in hit:
            continue

        row = {
            k: v[0] if isinstance(v, list) and len(v) == 1 else v
            for k, v in hit["fields"].items()
        }
        data.append(row)

    return data


def main():
    load_dotenv()

    ES_HOST = os.getenv("ES_HOST")
    ES_PORT = os.getenv("ES_PORT")
    ES_API_KEY = os.getenv("ES_API_KEY")

    ES_INDEX = os.getenv("ES_INDEX")

    if not all([ES_HOST, ES_PORT, ES_API_KEY]):
        missing = [key for key, value in {"ES_HOST": ES_HOST, "ES_PORT": ES_PORT, "ES_API_KEY": ES_API_KEY}.items() if
                   not value]
        print(f"Error: Missing environment variables: {missing}")
        exit(1)

    # Create the Elasticsearch client.
    elasticsearch_client = Elasticsearch(
        [f"https://{ES_HOST}:{ES_PORT}"],
        api_key=ES_API_KEY,
        verify_certs=False,
        ssl_show_warn=False,
        request_timeout=600,  # 10 minutes
        max_retries=3,
        retry_on_timeout=True
    )

    # Print basic info about the Elasticsearch cluster.
    print("Elasticsearch Cluster Info:")
    print((elasticsearch_client.info()))

    # Retrieve data from the source index.
    print(f"\nReading data from '{ES_INDEX}':")

    data = get_data_from_elastic(elasticsearch_client, ES_INDEX)
    print(f"Retrieved {len(data)} documents from Elasticsearch.")

    df = pd.DataFrame(data)

    # Save the Pandas DataFrame to disk in raw form
    # as CSV
    df.to_csv("../data/raw_vm_data_pandas.csv", index=False)
    # as Parquet
    df.to_parquet("../data/raw_vm_data_pandas.parquet", index=False)

    # Convert the list of dicts to a Polars DataFrame
    df_polars = pl.from_pandas(df)
    print("\nSample Polars DataFrame:")
    print(df_polars.head(2))

    # Save the Polars DataFrame to disk
    df_polars.write_parquet("../data/raw_vm_data_polars.parquet")


if __name__ == '__main__':
    main()
