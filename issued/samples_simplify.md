https://github.com/PyAutoLabs/PyAutoFit/issues/1002


Issue Simplify Samples calculation via search #1002 — Simplify samples coupling: The samples_cls / samples_via_internal_from / samples_via_csv_from methods are tightly coupled. A factory in Samples that inspects search_internal type would decouple sample types from search types.