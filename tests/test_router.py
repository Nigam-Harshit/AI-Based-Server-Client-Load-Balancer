"""Tests for routing algorithms."""

import unittest

from load_balancer.router import (
    IPHashRouter,
    LeastConnectionsRouter,
    RoundRobinRouter,
    get_router,
)


class TestRouterAlgorithms(unittest.TestCase):
    def setUp(self):
        self.backends = [
            "http://127.0.0.1:8001",
            "http://127.0.0.1:8002",
            "http://127.0.0.1:8003",
        ]

    def test_round_robin_sequential_cycling(self):
        """Verify S1 -> S2 -> S3 -> S1 -> S2 -> S3 sequence."""
        router = RoundRobinRouter(self.backends)
        expected_sequence = [
            self.backends[0],
            self.backends[1],
            self.backends[2],
            self.backends[0],
            self.backends[1],
            self.backends[2],
        ]
        actual_sequence = [router.select() for _ in range(6)]
        self.assertEqual(actual_sequence, expected_sequence)

    def test_least_connections_selection(self):
        """Construct state S1=3, S2=1, S3=2 and verify S2 is chosen."""
        router = LeastConnectionsRouter(self.backends)
        router.set_active_connections(self.backends[0], 3)
        router.set_active_connections(self.backends[1], 1)
        router.set_active_connections(self.backends[2], 2)

        # Selection should pick S2 (count=1) and increment its count to 2
        selected = router.select()
        self.assertEqual(selected, self.backends[1])
        counts = router.get_active_connections()
        self.assertEqual(counts[self.backends[1]], 2)

    def test_least_connections_equal_counts_tie_breaking(self):
        """Verify deterministic tie-breaking: first backend in configuration order."""
        router = LeastConnectionsRouter(self.backends)
        # All counts start at 0
        selected1 = router.select()
        # Should pick index 0 (S1) and increment S1 to 1
        self.assertEqual(selected1, self.backends[0])

        # Remaining S2 and S3 are both at 0; next selection should pick S2
        selected2 = router.select()
        self.assertEqual(selected2, self.backends[1])

        # S3 is at 0; next selection should pick S3
        selected3 = router.select()
        self.assertEqual(selected3, self.backends[2])

    def test_least_connections_increment_decrement_and_failure_release(self):
        """Verify counts increment on select and decrement on release, including exceptions."""
        router = LeastConnectionsRouter(self.backends)
        backend = self.backends[0]

        # Explicit select and release
        self.assertEqual(router.get_active_connections()[backend], 0)
        chosen = router.select()
        self.assertEqual(chosen, backend)
        self.assertEqual(router.get_active_connections()[backend], 1)
        router.release(chosen)
        self.assertEqual(router.get_active_connections()[backend], 0)

        # Context manager normal execution
        with router.route() as b:
            self.assertEqual(b, backend)
            self.assertEqual(router.get_active_connections()[backend], 1)
        self.assertEqual(router.get_active_connections()[backend], 0)

        # Context manager with exception
        try:
            with router.route() as b:
                self.assertEqual(router.get_active_connections()[b], 1)
                raise RuntimeError("Simulated request failure")
        except RuntimeError:
            pass

        # Count must be cleanly decremented back to 0 even after failure
        self.assertEqual(router.get_active_connections()[backend], 0)

    def test_ip_hash_deterministic_mapping(self):
        """Verify repeated selections for identical IP produce identical backend."""
        router = IPHashRouter(self.backends)
        client_ip = "192.168.1.10"

        first_selection = router.select(client_ip)
        for _ in range(5):
            self.assertEqual(router.select(client_ip), first_selection)

    def test_ip_hash_different_ips(self):
        """Verify different client IPs can be mapped to different servers."""
        router = IPHashRouter(self.backends)
        test_ips = ["10.0.0.1", "10.0.0.2", "192.168.1.50", "172.16.0.4", "127.0.0.1"]
        selections = {router.select(ip) for ip in test_ips}
        # Across multiple varied IPs, multiple distinct backends should be selected
        self.assertGreater(len(selections), 1)

    def test_get_router_factory(self):
        """Verify router factory creates proper router types and validates input."""
        rr = get_router("round_robin", self.backends)
        self.assertIsInstance(rr, RoundRobinRouter)

        lc = get_router("least_connections", self.backends)
        self.assertIsInstance(lc, LeastConnectionsRouter)

        iph = get_router("ip_hash", self.backends)
        self.assertIsInstance(iph, IPHashRouter)

        with self.assertRaises(ValueError):
            get_router("invalid_algo", self.backends)


if __name__ == "__main__":
    unittest.main()
