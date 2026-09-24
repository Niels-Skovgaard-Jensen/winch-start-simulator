from .app import main

# The guard matters: simulation workers are started with "spawn", which re-imports
# the main module in every worker process.
if __name__ == "__main__":
    main()
