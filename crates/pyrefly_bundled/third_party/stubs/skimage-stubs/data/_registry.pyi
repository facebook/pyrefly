# This minimal dataset was available as part of
# scikit-image 0.15 and will be retained until
# further notice.
# Testing data and additional datasets should only
# be made available by pooch
legacy_datasets: list = ...

# Registry of datafiles that can be downloaded along with their SHA256 hashes
# To generate the SHA256 hash, use the command
# openssl sha256 filename
registry: dict = ...

registry_urls: dict = ...

legacy_registry: dict = ...
