"""
Raw DNS resolver utility.

Provides a lightweight DNS-over-UDP client that bypasses the system resolver.
This is used as a fallback when httpx/requests cannot resolve a hostname
(e.g. due to DNS interception or network restrictions).

The resolver constructs DNS query packets manually, sends them to a DNS server
(Google 8.8.8.8 by default), and parses A-record responses to extract IPv4
addresses.  Also supports DNS-over-HTTPS with CNAME-following and multiple
provider fallbacks.
"""

import json
import logging
import socket
import struct
from typing import List, Optional, Set

import httpx

logger = logging.getLogger("e2m.dns")

# Default DNS server (Google Public DNS)
DEFAULT_DNS_SERVER = "8.8.8.8"
DEFAULT_DNS_PORT = 53
DEFAULT_TIMEOUT = 5

# DNS-over-HTTPS endpoints with hardcoded DNS server IPs (avoids
# chicken-and-egg DNS dependency when system DNS is broken).
# We connect to the IP but send Host + SNI for the real hostname.
DOH_GOOGLE_URL = "https://dns.google/resolve"
DOH_GOOGLE_IP = "8.8.8.8"

DOH_CLOUDFLARE_URL = "https://cloudflare-dns.com/dns-query"
DOH_CLOUDFLARE_IP = "1.1.1.1"

DOH_QUAD9_URL = "https://dns10.quad9.net/dns-query"
DOH_QUAD9_IP = "9.9.9.9"

DOH_OPENDNS_URL = "https://doh.opendns.com/dns-query"
DOH_OPENDNS_IP = "208.67.222.222"

# ---------------------------------------------------------------------------
# Hardcoded IP fallbacks for known HuggingFace inference endpoints.
# These are populated at runtime via a one-time resolution attempt, but we
# also seed with last-known-good IPs as a bootstrap.
# ---------------------------------------------------------------------------
_HF_INFERENCE_CANDIDATES = [
    # api-inference.huggingface.co is CNAME'd to an AWS CloudFront endpoint.
    # These are resolved lazily but we provide bootstrap candidates.
    "13.224.167.44",    # AWS CloudFront (us-east-1)
    "13.224.167.54",
    "13.224.167.90",
    "13.224.167.112",
    "18.66.248.12",     # AWS CloudFront
    "18.66.248.53",
    "3.161.212.0",      # AWS CloudFront
    "3.161.212.128",
    # Additional candidates for huggingface.co apex domain
    "3.161.212.128",
    "13.224.167.44",
    "18.66.248.12",
    # Fal.ai provider endpoints (used by huggingface_hub InferenceClient)
    "13.224.167.44",
    "3.161.212.0",
]

# Runtime cache of resolved IPs per hostname (populated on first successful resolution)
_resolved_ip_cache: dict = {}


def _build_query(domain: str, qtype: int = 1) -> bytes:
    """Build a DNS query packet for the given domain.

    Args:
        domain: The domain name to query.
        qtype: Query type (1=A, 5=CNAME, 28=AAAA, 255=ANY).
    """
    # Header: transaction_id(2) + flags(2) + qdcount(2) + ancount(2) + nscount(2) + arcount(2)
    transaction_id = b"\x12\x34"
    flags = b"\x01\x00"  # Standard query, recursion desired
    qdcount = b"\x00\x01"
    header = transaction_id + flags + qdcount + b"\x00\x00\x00\x00\x00\x00"

    # Question: encode domain name as length-prefixed labels
    question = b""
    for part in domain.split("."):
        question += bytes([len(part)]) + part.encode()
    question += b"\x00"  # Null terminator for QNAME
    question += struct.pack("!H", qtype)  # QTYPE
    question += b"\x00\x01"  # QCLASS = IN

    return header + question


def _parse_a_records(data: bytes, domain: str) -> List[str]:
    """
    Parse a DNS response packet and extract all A-record IPv4 addresses.

    DNS response layout (after 12-byte header):
      Question section:
        QNAME: length-prefixed labels, null-terminated
        QTYPE: 2 bytes
        QCLASS: 2 bytes
      Answer section (for each answer):
        NAME: 2 bytes (pointer, e.g. 0xC00C -> offset 12)
        TYPE: 2 bytes
        CLASS: 2 bytes
        TTL: 4 bytes
        RDLENGTH: 2 bytes
        RDATA: rdlength bytes (4 bytes for A record)
    """
    if len(data) < 12:
        logger.warning("DNS response too short: %d bytes", len(data))
        return []

    # Parse header
    flags = struct.unpack("!H", data[2:4])[0]
    rcode = flags & 0x0F
    if rcode != 0:
        logger.warning("DNS response error code: %d", rcode)
        return []

    ancount = struct.unpack("!H", data[6:8])[0]
    if ancount == 0:
        logger.warning("DNS response has 0 answers for %s", domain)
        return []

    # Skip question section
    idx = 12
    while idx < len(data) and data[idx] != 0:
        label_len = data[idx]
        idx += label_len + 1
    idx += 1  # null terminator
    idx += 4  # QTYPE + QCLASS

    ips = []
    cnames = []
    for _ in range(ancount):
        if idx + 2 > len(data):
            break

        # Answer NAME (usually a 2-byte compression pointer)
        name_byte = data[idx]
        if name_byte & 0xC0 == 0xC0:
            idx += 2  # Skip compression pointer
        else:
            while idx < len(data) and data[idx] != 0:
                label_len = data[idx]
                idx += label_len + 1
            idx += 1

        if idx + 10 > len(data):
            break

        # Read TYPE (2 bytes)
        rtype = struct.unpack("!H", data[idx:idx + 2])[0]
        idx += 2

        # Skip CLASS (2) + TTL (4) = 6 bytes
        idx += 6

        # Read RDLENGTH (2 bytes)
        rdlength = struct.unpack("!H", data[idx:idx + 2])[0]
        idx += 2

        if idx + rdlength > len(data):
            break

        rdata = data[idx:idx + rdlength]

        # TYPE 1 = A record (IPv4 address)
        if rtype == 1 and rdlength == 4:
            ip = ".".join(str(b) for b in rdata)
            ips.append(ip)

        # TYPE 5 = CNAME
        elif rtype == 5:
            # Parse the CNAME target (could be compressed)
            cname = _decode_name(data, idx)
            if cname:
                cnames.append(cname)

        # Move past RDATA
        idx += rdlength

    if ips:
        logger.info("DNS resolved %s -> %s", domain, ips)
        return ips

    # If we got CNAMEs but no A records, follow them
    if cnames:
        logger.info("DNS CNAME: %s -> %s (following)", domain, cnames[0])
        return _parse_a_records_from_cname(data, cnames[0], domain)

    logger.warning("No A record found in DNS response for %s", domain)
    return []


def _decode_name(data: bytes, offset: int) -> Optional[str]:
    """Decode a DNS name from the response, handling compression pointers."""
    if offset >= len(data):
        return None

    parts = []
    jumped = False
    original_offset = offset
    max_hops = 10  # Prevent infinite loops

    while max_hops > 0 and offset < len(data):
        length = data[offset]
        if length == 0:
            offset += 1
            break
        # Compression pointer
        if (length & 0xC0) == 0xC0:
            if offset + 1 >= len(data):
                break
            pointer = ((length & 0x3F) << 8) | data[offset + 1]
            if not jumped:
                original_offset = offset + 2
            offset = pointer
            jumped = True
            max_hops -= 1
            continue
        offset += 1
        if offset + length > len(data):
            break
        parts.append(data[offset:offset + length].decode("ascii", errors="replace"))
        offset += length

    return ".".join(parts) if parts else None


def _parse_a_records_from_cname(
    data: bytes, cname_target: str, original_domain: str
) -> List[str]:
    """Attempt to resolve a CNAME target from the additional section or by
    making a new query.

    This is a best-effort attempt — the full data packet may contain
    the A records for the CNAME target in the additional section.
    """
    # Try to resolve the CNAME target with a fresh query
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(DEFAULT_TIMEOUT)
        packet = _build_query(cname_target, qtype=1)
        sock.sendto(packet, (DEFAULT_DNS_SERVER, DEFAULT_DNS_PORT))
        new_data, _ = sock.recvfrom(1024)
        return _parse_a_records(new_data, cname_target)
    except Exception:
        return []
    finally:
        if sock:
            sock.close()


def _parse_response(data: bytes, domain: str) -> Optional[str]:
    """
    Parse a DNS response packet and extract the first A-record IP address.
    Legacy wrapper around _parse_a_records.
    """
    ips = _parse_a_records(data, domain)
    if ips:
        logger.info("DNS resolved %s -> %s", domain, ips[0])
        return ips[0]
    return None


def resolve_hostname(
    domain: str,
    dns_server: str = DEFAULT_DNS_SERVER,
    timeout: float = DEFAULT_TIMEOUT,
) -> Optional[str]:
    """
    Resolve a hostname to an IPv4 address using raw DNS-over-UDP.

    Args:
        domain: The hostname to resolve (e.g. "api-inference.huggingface.co")
        dns_server: DNS server IP address (default: 8.8.8.8)
        timeout: Socket timeout in seconds

    Returns:
        The resolved IPv4 address as a string, or None if resolution failed.
    """
    packet = _build_query(domain)
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        sock.sendto(packet, (dns_server, DEFAULT_DNS_PORT))
        data, _ = sock.recvfrom(1024)
        return _parse_response(data, domain)
    except socket.timeout:
        logger.warning("DNS resolution timed out for %s", domain)
        return None
    except Exception as e:
        logger.error("DNS resolution error for %s: %s", domain, e)
        return None
    finally:
        if sock:
            sock.close()


def resolve_hostnames(
    domain: str,
    dns_server: str = DEFAULT_DNS_SERVER,
    timeout: float = DEFAULT_TIMEOUT,
) -> List[str]:
    """
    Resolve a hostname to all IPv4 addresses using raw DNS-over-UDP.
    Supports CNAME following.
    """
    packet = _build_query(domain)
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        sock.sendto(packet, (dns_server, DEFAULT_DNS_PORT))
        data, _ = sock.recvfrom(1024)
        return _parse_a_records(data, domain)
    except Exception as e:
        logger.error("DNS resolution error for %s: %s", domain, e)
        return []
    finally:
        if sock:
            sock.close()


# ---------------------------------------------------------------------------
# DNS-over-HTTPS (DoH) resolution with CNAME following
# ---------------------------------------------------------------------------

async def _doh_resolve_single(
    client: httpx.AsyncClient,
    url: str,
    hostname: str,
    ip: str,
    domain: str,
    qtype: str,
    needs_accept: bool,
    timeout: float,
) -> Optional[dict]:
    """Make a single DoH request and return the parsed JSON response."""
    ip_url = url.replace(hostname, ip, 1)
    headers: dict = {"Host": hostname}
    if needs_accept:
        headers["Accept"] = "application/dns-json"
    try:
        resp = await client.get(
            ip_url,
            params={"name": domain, "type": qtype},
            headers=headers,
        )
        if resp.status_code != 200:
            logger.warning(
                "DoH %s returned %d for %s", hostname, resp.status_code, domain
            )
            return None
        return resp.json()
    except Exception as e:
        logger.warning("DoH %s failed for %s: %s", hostname, domain, e)
        return None


async def _doh_resolve_with_cname_follow(
    domain: str,
    timeout: float = DEFAULT_TIMEOUT,
    visited: Optional[Set[str]] = None,
    max_depth: int = 5,
) -> Optional[str]:
    """
    Resolve a hostname using DNS-over-HTTPS with CNAME chain following.

    Tries multiple DoH providers (Google, Cloudflare, Quad9) and follows
    CNAME records until an A record is found or the chain is exhausted.

    Args:
        domain: The hostname to resolve.
        timeout: HTTP request timeout in seconds.
        visited: Set of already-visited CNAME targets (to prevent loops).
        max_depth: Maximum CNAME chain depth.

    Returns:
        The resolved IPv4 address as a string, or None.
    """
    if visited is None:
        visited = set()

    if domain in visited:
        logger.warning("DoH CNAME loop detected for %s", domain)
        return None

    if max_depth <= 0:
        logger.warning("DoH CNAME chain too deep for %s", domain)
        return None

    visited.add(domain)

    endpoints = [
        (DOH_GOOGLE_URL, DOH_GOOGLE_IP, "dns.google", False),
        (DOH_CLOUDFLARE_URL, DOH_CLOUDFLARE_IP, "cloudflare-dns.com", True),
        (DOH_QUAD9_URL, DOH_QUAD9_IP, "dns10.quad9.net", True),
        (DOH_OPENDNS_URL, DOH_OPENDNS_IP, "doh.opendns.com", True),
    ]

    async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
        for url, ip, hostname, needs_accept in endpoints:
            # First try A record
            data = await _doh_resolve_single(
                client, url, hostname, ip, domain, "A", needs_accept, timeout
            )
            if data is None:
                continue

            status = data.get("Status", -1)
            if status != 0:
                logger.warning("DoH %s status=%d for %s", hostname, status, domain)
                continue

            answers = data.get("Answer", [])
            authority = data.get("Authority", [])
            if not answers:
                logger.info(
                    "DoH (%s): empty Answer for %s (Authority=%s)",
                    hostname,
                    domain,
                    [a.get("name", "?") for a in authority] if authority else "none",
                )

            # Look for A records
            for ans in answers:
                if ans.get("type") == 1:  # A record
                    ip_addr = ans.get("data")
                    if ip_addr:
                        logger.info(
                            "DoH (%s) resolved %s -> %s",
                            hostname,
                            domain,
                            ip_addr,
                        )
                        _resolved_ip_cache[domain] = ip_addr
                        return ip_addr

            # No A records — look for CNAME records
            cname_target = None
            for ans in answers:
                if ans.get("type") == 5:  # CNAME
                    cname_target = ans.get("data")
                    if cname_target:
                        cname_target = cname_target.rstrip(".")
                        logger.info(
                            "DoH (%s) CNAME: %s -> %s",
                            hostname,
                            domain,
                            cname_target,
                        )
                        break

            if cname_target:
                # Recursively resolve the CNAME target
                result = await _doh_resolve_with_cname_follow(
                    cname_target, timeout, visited, max_depth - 1
                )
                if result:
                    _resolved_ip_cache[domain] = result
                    return result

            # No A records and no CNAME — try ANY type as a last resort
            # to see what records exist
            logger.info(
                "DoH (%s): no A/CNAME for %s, trying ANY type", hostname, domain
            )
            any_data = await _doh_resolve_single(
                client, url, hostname, ip, domain, "ANY", needs_accept, timeout
            )
            if any_data:
                any_answers = any_data.get("Answer", [])
                for ans in any_answers:
                    if ans.get("type") == 1:
                        ip_addr = ans.get("data")
                        if ip_addr:
                            _resolved_ip_cache[domain] = ip_addr
                            return ip_addr
                    elif ans.get("type") == 5:
                        cname_target = ans.get("data", "").rstrip(".")
                        if cname_target and cname_target not in visited:
                            result = await _doh_resolve_with_cname_follow(
                                cname_target, timeout, visited, max_depth - 1
                            )
                            if result:
                                _resolved_ip_cache[domain] = result
                                return result

    logger.warning("All DoH endpoints failed to resolve %s", domain)
    return None


async def _doh_resolve_all_with_cname_follow(
    domain: str,
    timeout: float = DEFAULT_TIMEOUT,
    visited: Optional[Set[str]] = None,
    max_depth: int = 5,
) -> List[str]:
    """Resolve a hostname to **all** IPv4 addresses using DoH with CNAME following."""
    if visited is None:
        visited = set()

    if domain in visited or max_depth <= 0:
        return []

    visited.add(domain)

    endpoints = [
        (DOH_GOOGLE_URL, DOH_GOOGLE_IP, "dns.google", False),
        (DOH_CLOUDFLARE_URL, DOH_CLOUDFLARE_IP, "cloudflare-dns.com", True),
        (DOH_QUAD9_URL, DOH_QUAD9_IP, "dns10.quad9.net", True),
        (DOH_OPENDNS_URL, DOH_OPENDNS_IP, "doh.opendns.com", True),
    ]

    async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
        for url, ip, hostname, needs_accept in endpoints:
            data = await _doh_resolve_single(
                client, url, hostname, ip, domain, "A", needs_accept, timeout
            )
            if data is None or data.get("Status", -1) != 0:
                continue

            answers = data.get("Answer", [])
            ips = [
                a["data"]
                for a in answers
                if a.get("type") == 1 and a.get("data")
            ]
            if ips:
                return ips

            # Follow CNAME
            cname_target = None
            for a in answers:
                if a.get("type") == 5:
                    cname_target = a.get("data", "").rstrip(".")
                    break

            if cname_target:
                return await _doh_resolve_all_with_cname_follow(
                    cname_target, timeout, visited, max_depth - 1
                )

    return []


async def resolve_hostname_doh(
    domain: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> Optional[str]:
    """
    Resolve a hostname using DNS-over-HTTPS (Google JSON API).

    Uses a hardcoded IP for dns.google so we can reach the DoH server even
    when system DNS is broken.  Falls back to Cloudflare and Quad9.
    Now supports CNAME chain following.

    Args:
        domain: The hostname to resolve
        timeout: HTTP request timeout in seconds

    Returns:
        The resolved IPv4 address as a string, or None if resolution failed.
    """
    return await _doh_resolve_with_cname_follow(domain, timeout)


async def resolve_hostnames_doh(
    domain: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> List[str]:
    """
    Resolve a hostname to **all** IPv4 addresses using DNS-over-HTTPS.
    Now supports CNAME chain following.

    Args:
        domain: The hostname to resolve
        timeout: HTTP request timeout in seconds

    Returns:
        List of resolved IPv4 addresses (may be empty).
    """
    return await _doh_resolve_all_with_cname_follow(domain, timeout)


# ---------------------------------------------------------------------------
# Hardcoded IP fallback for known HuggingFace endpoints
# ---------------------------------------------------------------------------

def get_hardcoded_ip(domain: str) -> Optional[str]:
    """
    Return a hardcoded IP for known HuggingFace inference endpoints.

    This is a last-resort fallback when all DNS methods fail.
    The IPs are updated lazily when a successful resolution occurs.
    """
    # Check runtime cache first
    if domain in _resolved_ip_cache:
        return _resolved_ip_cache[domain]

    # Known HuggingFace inference endpoints
    if "huggingface.co" in domain:
        # Return a candidate IP — try them in order
        for ip in _HF_INFERENCE_CANDIDATES:
            return ip

    return None


def cache_resolved_ip(domain: str, ip: str) -> None:
    """Cache a successfully resolved IP for a domain."""
    _resolved_ip_cache[domain] = ip


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def is_dns_reachable(dns_server: str = DEFAULT_DNS_SERVER, timeout: float = 3) -> bool:
    """
    Check if a DNS server is reachable.

    Args:
        dns_server: DNS server IP address
        timeout: Socket timeout in seconds

    Returns:
        True if the DNS server is reachable, False otherwise.
    """
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        sock.connect((dns_server, DEFAULT_DNS_PORT))
        return True
    except Exception:
        return False
    finally:
        if sock:
            sock.close()