# Divvy adaptation changes

1. 5 min -> 10 min; 12->12 -> 6->6.
2. Removed hard-coded `A_t[:-(24-1)]`; alignment is represented explicitly by `graph_id`, with boundary loss `6+6-1=11`.
3. Removed hard-coded physical fleet total `494`; `fleet_size` comes from V7 mass conservation.
4. Model nodes are `[0] + V7 station_ids`.
5. Dynamic graph storage changed from dense to sparse packed format for Divvy scale.
6. Dynamic graphs use V7 physical station<->0 user transitions plus station->station relocation transitions.
7. Static support is training-only to reduce topology leakage.
8. DataLoader carries `graph_id`, so shuffled training remains exactly aligned with dynamic graphs.
9. Zero bike count is treated as valid; Divvy config uses unmasked metrics/loss.
10. Device selection falls back to CPU automatically if CUDA is unavailable.
