import time
import itertools
import random
from typing import List, Tuple, Optional, Dict
from Crypto.Cipher import DES


class DESCracker:
    """
    Educational DES cryptanalysis tool implementing various attack methods
    """
    
    def __init__(self):
        self.block_size = 8  # DES block size in bytes
        
    def _is_valid_plaintext(self, data: bytes, expected_plaintext: bytes = None) -> bool:
        """
        Check if decrypted data matches expected plaintext
        
        Args:
            data: Decrypted data to validate
            expected_plaintext: Expected plaintext (if known)
            
        Returns:
            bool: True if data matches expected plaintext or appears valid
        """
        # If we have expected plaintext, check exact match
        if expected_plaintext is not None:
            return data == expected_plaintext
            
        # If no expected plaintext, check for printable ASCII characters (simple heuristic)
        try:
            decoded = data.decode('ascii')
            return all(32 <= ord(c) <= 126 or c in '\n\r\t' for c in decoded)
        except (UnicodeDecodeError, AttributeError):
            return False
    
    def _generate_reduced_keys(self, key_length: int) -> itertools.product:
        """Generate reduced keyspace for demo mode"""
        if key_length <= 3:
            return itertools.product(range(256), repeat=key_length)
        else:
            common_bytes = [0x00, 0x01, 0x10, 0x11, 0x20, 0x55, 0xAA, 0xFF]
            return itertools.product(common_bytes, repeat=key_length)
    
    def _generate_full_keys(self) -> itertools.product:
        """Generate full DES keyspace (56-bit effective)"""
        # Generate all possible 8-byte keys
        return itertools.product(range(256), repeat=8)
    
    def _pad_key_to_des_length(self, short_key: bytes) -> bytes:
        """
        Pad a short key to DES key length (8 bytes)
        
        Args:
            short_key: Short key to pad
            
        Returns:
            8-byte DES key
        """
        if len(short_key) >= 8:
            return short_key[:8]
        padded = short_key + b'\x00' * (8 - len(short_key))
        return padded
    
    def _estimate_time_remaining(self, attempts: int, total_keyspace: int, elapsed: float) -> str:
        """Estimate remaining time based on current progress"""
        if attempts == 0 or elapsed == 0:
            return "Unknown"
        
        rate = attempts / elapsed  # keys per second
        remaining_keys = total_keyspace - attempts
        if rate == 0:
            return "Infinite"
            
        remaining_seconds = remaining_keys / rate
        
        if remaining_seconds < 60:
            return f"{remaining_seconds:.1f} seconds"
        elif remaining_seconds < 3600:
            return f"{remaining_seconds/60:.1f} minutes"
        elif remaining_seconds < 86400:
            return f"{remaining_seconds/3600:.1f} hours"
        elif remaining_seconds < 31536000:
            return f"{remaining_seconds/86400:.1f} days"
        else:
            return f"{remaining_seconds/31536000:.1f} years"
    
    def single_des_bruteforce_demo(self, ciphertext: bytes, known_plaintext: bytes = None) -> Optional[Tuple[bytes, bytes]]:
        """Demo version with reduced keyspace"""
        print("Starting Single DES brute-force (DEMO MODE)...")
        start_time = time.time()
        attempts = 0
        
        for key_length in range(1, 4):
            print(f"Trying keys of length {key_length}...")
            for key_tuple in self._generate_reduced_keys(key_length):
                attempts += 1
                short_key = bytes(key_tuple)
                des_key = self._pad_key_to_des_length(short_key)
                
                try:
                    cipher = DES.new(des_key, DES.MODE_ECB)
                    decrypted = cipher.decrypt(ciphertext)
                    
                    if self._is_valid_plaintext(decrypted, known_plaintext):
                        elapsed = time.time() - start_time
                        print(f"SUCCESS! Key: {short_key.hex()}")
                        print(f"Attempts: {attempts}, Time: {elapsed:.2f}s")
                        print(f"Decrypted: {decrypted}")
                        return des_key, decrypted
                        
                except Exception as e:
                    continue
                
                if attempts % 1000 == 0:
                    print(f"Tried {attempts} keys...")
        
        elapsed = time.time() - start_time
        print(f"Failed after {attempts} attempts in {elapsed:.2f}s")
        return None
    
    def single_des_bruteforce_real(self, ciphertext: bytes, known_plaintext: bytes = None) -> Optional[Tuple[bytes, bytes]]:
        """Real version with full keyspace"""
        print("Starting Single DES brute-force (REAL MODE)...")
        print("WARNING: This may take hours to years depending on key position!")
        print("Full keyspace: 2^56 = 72,057,594,037,927,936 keys")
        
        start_time = time.time()
        attempts = 0
        total_keyspace = 2**56  # Theoretical, actual is 2^64 but we estimate
        
        # Start with random position to avoid worst case
        start_key = random.randbytes(8)
        print(f"Starting from random key: {start_key.hex()}")
        
        # Generate keys starting from random position
        key_int = int.from_bytes(start_key, 'big')
        
        while attempts < total_keyspace:
            attempts += 1
            
            # Convert current key_int to bytes
            try:
                current_key = (key_int % (2**64)).to_bytes(8, 'big')
                key_int += 1
                
                cipher = DES.new(current_key, DES.MODE_ECB)
                decrypted = cipher.decrypt(ciphertext)
                
                if self._is_valid_plaintext(decrypted, known_plaintext):
                    elapsed = time.time() - start_time
                    print(f"\nSUCCESS! Key: {current_key.hex()}")
                    print(f"Attempts: {attempts:,}, Time: {elapsed:.2f}s")
                    print(f"Decrypted: {decrypted}")
                    return current_key, decrypted
                    
            except Exception as e:
                key_int += 1
                continue
            
            # Progress reporting every 1M attempts
            if attempts % 1000000 == 0:
                elapsed = time.time() - start_time
                rate = attempts / elapsed if elapsed > 0 else 0
                remaining = self._estimate_time_remaining(attempts, total_keyspace, elapsed)
                print(f"Progress: {attempts:,} keys ({rate:.0f} keys/sec) - ETA: {remaining}")
        
        elapsed = time.time() - start_time
        print(f"Exhausted keyspace after {attempts:,} attempts in {elapsed:.2f}s")
        return None
    
    def double_des_meet_in_middle_demo(self, ciphertext: bytes, known_plaintext: bytes) -> Optional[Tuple[bytes, bytes, bytes]]:
        """Demo version of Double DES meet-in-the-middle attack"""
        print("Starting Double DES meet-in-the-middle (DEMO MODE)...")
        start_time = time.time()
        
        # Phase 1: Build encryption table
        print("Phase 1: Building encryption table...")
        encryption_table: Dict[bytes, List[bytes]] = {}
        build_attempts = 0
        max_table_size = 10000  # Small size for demo
        
        for key_length in range(1, 3):
            for key_tuple in self._generate_reduced_keys(key_length):
                if build_attempts >= max_table_size:
                    break
                    
                build_attempts += 1
                short_key = bytes(key_tuple)
                des_key = self._pad_key_to_des_length(short_key)
                
                try:
                    cipher1 = DES.new(des_key, DES.MODE_ECB)
                    intermediate = cipher1.encrypt(known_plaintext)
                    if intermediate not in encryption_table:
                        encryption_table[intermediate] = []
                    encryption_table[intermediate].append(des_key)
                except Exception:
                    continue
            
            if build_attempts >= max_table_size:
                break
        
        print(f"Phase 1 complete: {len(encryption_table):,} unique intermediates from {build_attempts} keys")
        
        # Phase 2: Search for matches
        print("Phase 2: Searching for matches...")
        search_attempts = 0
        max_search = 10000  # Limit search attempts for demo
        
        for key_length in range(1, 3):
            for key_tuple in self._generate_reduced_keys(key_length):
                if search_attempts >= max_search:
                    break
                    
                search_attempts += 1
                short_key2 = bytes(key_tuple)
                des_key2 = self._pad_key_to_des_length(short_key2)
                
                try:
                    cipher2 = DES.new(des_key2, DES.MODE_ECB)
                    intermediate = cipher2.decrypt(ciphertext)
                    
                    if intermediate in encryption_table:
                        for des_key1 in encryption_table[intermediate]:
                            # Verify
                            cipher1 = DES.new(des_key1, DES.MODE_ECB)
                            cipher2_verify = DES.new(des_key2, DES.MODE_ECB)
                            temp = cipher2_verify.decrypt(ciphertext)
                            plaintext = cipher1.decrypt(temp)
                            
                            if plaintext == known_plaintext:
                                elapsed = time.time() - start_time
                                print(f"\nSUCCESS! Key1: {des_key1.hex()}")
                                print(f"Key2: {des_key2.hex()}")
                                print(f"Search attempts: {search_attempts:,}, Total time: {elapsed:.2f}s")
                                print(f"Decrypted: {plaintext}")
                                return des_key1, des_key2, plaintext
                        
                except Exception:
                    continue
            
            if search_attempts >= max_search:
                break
        
        elapsed = time.time() - start_time
        print(f"Demo completed without success after {search_attempts:,} search attempts in {elapsed:.2f}s")
        return None
    
    def double_des_meet_in_middle_real(self, ciphertext: bytes, known_plaintext: bytes) -> Optional[Tuple[bytes, bytes, bytes]]:
        """Real Double DES meet-in-the-middle attack"""
        print("Starting Double DES meet-in-the-middle (REAL MODE)...")
        print("WARNING: This requires ~72 petabytes of memory and days of computation!")
        print("Keyspace: 2^112, but reduced to 2^57 with meet-in-the-middle")
        
        start_time = time.time()
        
        # Phase 1: Build encryption table - this will be HUGE
        print("Phase 1: Building encryption table...")
        print("Note: This is a limited simulation of real attack")
        
        encryption_table: Dict[bytes, List[bytes]] = {}
        build_attempts = 0
        max_table_size = 1000000  # Limit for memory reasons
        
        # Use random sampling instead of full keyspace
        while build_attempts < max_table_size:
            build_attempts += 1
            random_key = random.randbytes(8)
            
            try:
                cipher1 = DES.new(random_key, DES.MODE_ECB)
                intermediate = cipher1.encrypt(known_plaintext)
                if intermediate not in encryption_table:
                    encryption_table[intermediate] = []
                encryption_table[intermediate].append(random_key)
            except Exception:
                continue
            
            if build_attempts % 100000 == 0:
                elapsed = time.time() - start_time
                print(f"Built {build_attempts:,} entries in {elapsed:.1f}s...")
        
        print(f"Phase 1 complete: {len(encryption_table):,} unique intermediates")
        
        # Phase 2: Search for matches
        print("Phase 2: Searching for matches...")
        search_attempts = 0
        max_search = 10000000  # Limit search attempts
        
        while search_attempts < max_search:
            search_attempts += 1
            random_key2 = random.randbytes(8)
            
            try:
                cipher2 = DES.new(random_key2, DES.MODE_ECB)
                intermediate = cipher2.decrypt(ciphertext)
                
                if intermediate in encryption_table:
                    for des_key1 in encryption_table[intermediate]:
                        # Verify
                        cipher1 = DES.new(des_key1, DES.MODE_ECB)
                        cipher2_verify = DES.new(random_key2, DES.MODE_ECB)
                        temp = cipher2_verify.decrypt(ciphertext)
                        plaintext = cipher1.decrypt(temp)
                        
                        if plaintext == known_plaintext:
                            elapsed = time.time() - start_time
                            print(f"\nSUCCESS! Key1: {des_key1.hex()}")
                            print(f"Key2: {random_key2.hex()}")
                            print(f"Search attempts: {search_attempts:,}, Total time: {elapsed:.2f}s")
                            print(f"Decrypted: {plaintext}")
                            return des_key1, random_key2, plaintext
                        
            except Exception:
                continue
            
            if search_attempts % 1000000 == 0:
                elapsed = time.time() - start_time
                # Fixed rate calculation
                build_time = max_table_size * elapsed / (build_attempts + search_attempts)
                search_time = elapsed - build_time
                rate = search_attempts / search_time if search_time > 0 else 0
                print(f"Searched {search_attempts:,} keys ({rate:.0f} keys/sec)...")
        
        elapsed = time.time() - start_time
        print(f"Search phase completed without success after {search_attempts:,} attempts in {elapsed:.2f}s")
        return None
    
    def triple_des_bruteforce_demo(self, ciphertext: bytes, known_plaintext: bytes) -> Optional[Tuple[bytes, bytes, bytes, bytes]]:
        """Demo version of Triple DES brute-force attack"""
        print("Starting Triple DES brute-force (DEMO MODE)...")
        start_time = time.time()
        attempts = 0
        
        # Use specific demo keys that will be found quickly
        demo_keys = [
            b'\x00\x00\x00\x00\x00\x00\x00\x00',
            b'\x01\x00\x00\x00\x00\x00\x00\x00', 
            b'\x02\x00\x00\x00\x00\x00\x00\x00',
            b'\x01\x01\x00\x00\x00\x00\x00\x00',
            b'\x02\x01\x00\x00\x00\x00\x00\x00',
            b'\x01\x02\x00\x00\x00\x00\x00\x00'
        ]
        
        total_combinations = len(demo_keys) ** 3
        print(f"Trying {total_combinations:,} key combinations...")
        
        for key1 in demo_keys:
            for key2 in demo_keys:
                for key3 in demo_keys:
                    attempts += 1
                    
                    try:
                        # 3DES-EDE decryption: P = D_K3(E_K2(D_K1(C)))
                        cipher1 = DES.new(key1, DES.MODE_ECB)
                        cipher2 = DES.new(key2, DES.MODE_ECB)
                        cipher3 = DES.new(key3, DES.MODE_ECB)
                        
                        temp1 = cipher1.decrypt(ciphertext)
                        temp2 = cipher2.encrypt(temp1)
                        plaintext = cipher3.decrypt(temp2)
                        
                        if plaintext == known_plaintext:
                            elapsed = time.time() - start_time
                            print(f"\nSUCCESS! Found keys:")
                            print(f"Key1: {key1.hex()}")
                            print(f"Key2: {key2.hex()}")
                            print(f"Key3: {key3.hex()}")
                            print(f"Attempts: {attempts:,}, Time: {elapsed:.2f}s")
                            print(f"Decrypted: {plaintext}")
                            return key1, key2, key3, plaintext
                            
                    except Exception:
                        continue
                    
                    if attempts % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = attempts / elapsed if elapsed > 0 else 0
                        progress = (attempts / total_combinations) * 100
                        print(f"Progress: {attempts:,}/{total_combinations:,} ({progress:.1f}%) - {rate:.0f} attempts/sec")
        
        elapsed = time.time() - start_time
        print(f"Demo completed without success after {attempts:,} attempts in {elapsed:.2f}s")
        return None
    
    def triple_des_bruteforce_real(self, ciphertext: bytes, known_plaintext: bytes) -> Optional[Tuple[bytes, bytes, bytes, bytes]]:
        """Real Triple DES brute-force attack"""
        print("Starting Triple DES brute-force (REAL MODE)...")
        print("WARNING: This is computationally infeasible!")
        print("Keyspace: 2^168 = 374,144,419,156,711,147,060,143,317,175,368,453,031,918,731,001,856 keys")
        print("Estimated time: Longer than age of universe with current technology")
        
        start_time = time.time()
        attempts = 0
        max_attempts = 100000000  # 100M limit for demonstration
        
        print(f"Running limited simulation ({max_attempts:,} attempts)...")
        
        while attempts < max_attempts:
            attempts += 1
            
            # Generate random key triplet
            key1 = random.randbytes(8)
            key2 = random.randbytes(8)
            key3 = random.randbytes(8)
            
            try:
                # 3DES-EDE decryption: P = D_K3(E_K2(D_K1(C)))
                cipher1 = DES.new(key1, DES.MODE_ECB)
                cipher2 = DES.new(key2, DES.MODE_ECB)
                cipher3 = DES.new(key3, DES.MODE_ECB)
                
                temp1 = cipher1.decrypt(ciphertext)
                temp2 = cipher2.encrypt(temp1)
                plaintext = cipher3.decrypt(temp2)
                
                if plaintext == known_plaintext:
                    elapsed = time.time() - start_time
                    print(f"\nMIRACLE! Found keys:")
                    print(f"Key1: {key1.hex()}")
                    print(f"Key2: {key2.hex()}")
                    print(f"Key3: {key3.hex()}")
                    print(f"Attempts: {attempts:,}, Time: {elapsed:.2f}s")
                    print(f"Decrypted: {plaintext}")
                    return key1, key2, key3, plaintext
                    
            except Exception:
                continue
            
            if attempts % 10000000 == 0:
                elapsed = time.time() - start_time
                rate = attempts / elapsed if elapsed > 0 else 0
                remaining_attempts = max_attempts - attempts
                remaining_time = remaining_attempts / rate if rate > 0 else float('inf')
                print(f"Progress: {attempts:,}/{max_attempts:,} ({rate:.0f} attempts/sec) - ETA: {remaining_time/60:.1f} min")
        
        elapsed = time.time() - start_time
        print(f"Simulation completed without success after {attempts:,} attempts in {elapsed:.2f}s")
        print("This demonstrates why Triple DES is still considered secure")
        return None


class TestVectorGenerator:
    """
    Generate test vectors for demonstrating the attacks
    """
    
    @staticmethod
    def generate_single_des_test():
        """Generate test vector for Single DES"""
        plaintext = b"HELLO123"  # 8 bytes for DES block
        # Use a simple key for demonstration
        key = b"\x01\x02\x00\x00\x00\x00\x00\x00"  # Short key padded
        
        cipher = DES.new(key, DES.MODE_ECB)
        ciphertext = cipher.encrypt(plaintext)
        
        return {
            'plaintext': plaintext,
            'key': key,
            'ciphertext': ciphertext,
            'description': 'Single DES test'
        }
    
    @staticmethod
    def generate_double_des_test():
        """Generate test vector for Double DES"""
        plaintext = b"TESTDATA"  # 8 bytes
        key1 = b"\x01\x00\x00\x00\x00\x00\x00\x00"  # Simple key1
        key2 = b"\x02\x00\x00\x00\x00\x00\x00\x00"  # Simple key2
        
        # Double DES encryption: C = E_K2(E_K1(P))
        cipher1 = DES.new(key1, DES.MODE_ECB)
        cipher2 = DES.new(key2, DES.MODE_ECB)
        
        intermediate = cipher1.encrypt(plaintext)
        ciphertext = cipher2.encrypt(intermediate)
        
        return {
            'plaintext': plaintext,
            'key1': key1,
            'key2': key2,
            'ciphertext': ciphertext,
            'description': 'Double DES test'
        }
    
    @staticmethod
    def generate_triple_des_test():
        """Generate test vector for Triple DES"""
        plaintext = b"3DESTEST"  # 8 bytes
        key1 = b"\x01\x00\x00\x00\x00\x00\x00\x00"
        key2 = b"\x02\x00\x00\x00\x00\x00\x00\x00"
        key3 = b"\x01\x00\x00\x00\x00\x00\x00\x00"  # Same as key1 (common pattern)
        
        # 3DES-EDE encryption: C = E_K3(D_K2(E_K1(P)))
        cipher1 = DES.new(key1, DES.MODE_ECB)
        cipher2 = DES.new(key2, DES.MODE_ECB)
        cipher3 = DES.new(key3, DES.MODE_ECB)
        
        temp1 = cipher1.encrypt(plaintext)
        temp2 = cipher2.decrypt(temp1)
        ciphertext = cipher3.encrypt(temp2)
        
        return {
            'plaintext': plaintext,
            'key1': key1,
            'key2': key2,
            'key3': key3,
            'ciphertext': ciphertext,
            'description': 'Triple DES test'
        }
    
    @staticmethod
    def generate_real_test_vectors():
        """Generate test vectors with random keys for real attacks"""
        plaintext = b"REALTEST"
        
        # Single DES with random key
        single_key = random.randbytes(8)
        cipher = DES.new(single_key, DES.MODE_ECB)
        single_ciphertext = cipher.encrypt(plaintext)
        
        # Double DES with random keys
        double_key1 = random.randbytes(8)
        double_key2 = random.randbytes(8)
        cipher1 = DES.new(double_key1, DES.MODE_ECB)
        cipher2 = DES.new(double_key2, DES.MODE_ECB)
        intermediate = cipher1.encrypt(plaintext)
        double_ciphertext = cipher2.encrypt(intermediate)
        
        # Triple DES with random keys
        triple_key1 = random.randbytes(8)
        triple_key2 = random.randbytes(8)
        triple_key3 = random.randbytes(8)
        cipher1 = DES.new(triple_key1, DES.MODE_ECB)
        cipher2 = DES.new(triple_key2, DES.MODE_ECB)
        cipher3 = DES.new(triple_key3, DES.MODE_ECB)
        temp1 = cipher1.encrypt(plaintext)
        temp2 = cipher2.decrypt(temp1)
        triple_ciphertext = cipher3.encrypt(temp2)
        
        return {
            'plaintext': plaintext,
            'single': {'key': single_key, 'ciphertext': single_ciphertext},
            'double': {'key1': double_key1, 'key2': double_key2, 'ciphertext': double_ciphertext},
            'triple': {'key1': triple_key1, 'key2': triple_key2, 'key3': triple_key3, 'ciphertext': triple_ciphertext}
        }


def show_menu():
    print("\n" + "="*50)
    print("DES CRACKER - Choose Attack Mode")
    print("="*50)
    print("1. Demo Mode (Fast, reduced keyspace)")
    print("2. Real Single DES Attack (Hours to years)")
    print("3. Real Double DES Attack (Days, needs huge memory)")
    print("4. Real Triple DES Attack (Infeasible, demonstration only)")
    print("5. Exit")
    print("="*50)


def run_demo_mode():
    print("\nDEMO MODE - Educational demonstration with reduced keyspace")
    print("="*60)
    
    cracker = DESCracker()
    generator = TestVectorGenerator()
    
    # Test Single DES
    print("\n1. SINGLE DES ATTACK (DEMO)")
    test_single = generator.generate_single_des_test()
    print(f"Plaintext: {test_single['plaintext']}")
    print(f"Key: {test_single['key'].hex()}")
    print(f"Ciphertext: {test_single['ciphertext'].hex()}")
    
    result = cracker.single_des_bruteforce_demo(
        test_single['ciphertext'], 
        test_single['plaintext']
    )
    
    # Test Double DES (demo version)
    print("\n2. DOUBLE DES MEET-IN-THE-MIDDLE (DEMO)")
    test_double = generator.generate_double_des_test()
    print(f"Plaintext: {test_double['plaintext']}")
    print(f"Key1: {test_double['key1'].hex()}")
    print(f"Key2: {test_double['key2'].hex()}")
    print(f"Ciphertext: {test_double['ciphertext'].hex()}")
    
    result = cracker.double_des_meet_in_middle_demo(
        test_double['ciphertext'],
        test_double['plaintext']
    )

    # Test Triple DES (demo version)
    print("\n3. TRIPLE DES ATTACK (DEMO)")
    test_triple = generator.generate_triple_des_test()
    print(f"Plaintext: {test_triple['plaintext']}")
    print(f"Key1: {test_triple['key1'].hex()}")
    print(f"Key2: {test_triple['key2'].hex()}")
    print(f"Key3: {test_triple['key3'].hex()}")
    print(f"Ciphertext: {test_triple['ciphertext'].hex()}")
    
    result = cracker.triple_des_bruteforce_demo(
        test_triple['ciphertext'],
        test_triple['plaintext']
    )


def run_real_single_des():
    """Run real Single DES attack"""
    print("\nREAL SINGLE DES ATTACK")
    print("WARNING: This may take a very long time!")
    
    cracker = DESCracker()
    generator = TestVectorGenerator()
    real_tests = generator.generate_real_test_vectors()
    
    print(f"Target plaintext: {real_tests['plaintext']}")
    print(f"Target key: {real_tests['single']['key'].hex()}")
    print(f"Ciphertext: {real_tests['single']['ciphertext'].hex()}")
    
    result = cracker.single_des_bruteforce_real(
        real_tests['single']['ciphertext'],
        real_tests['plaintext']
    )
    return result


def run_real_double_des():
    """Run real Double DES attack"""
    print("\nREAL DOUBLE DES ATTACK")
    print("WARNING: This requires enormous memory and computation!")
    
    cracker = DESCracker()
    generator = TestVectorGenerator()
    real_tests = generator.generate_real_test_vectors()
    
    print(f"Target plaintext: {real_tests['plaintext']}")
    print(f"Target key1: {real_tests['double']['key1'].hex()}")
    print(f"Target key2: {real_tests['double']['key2'].hex()}")
    print(f"Ciphertext: {real_tests['double']['ciphertext'].hex()}")
    
    result = cracker.double_des_meet_in_middle_real(
        real_tests['double']['ciphertext'],
        real_tests['plaintext']
    )
    return result


def run_real_triple_des():
    """Run real Triple DES attack"""
    print("\nREAL TRIPLE DES ATTACK")
    print("WARNING: This is computationally infeasible!")
    
    cracker = DESCracker()
    generator = TestVectorGenerator()
    real_tests = generator.generate_real_test_vectors()
    
    print(f"Target plaintext: {real_tests['plaintext']}")
    print(f"Target key1: {real_tests['triple']['key1'].hex()}")
    print(f"Target key2: {real_tests['triple']['key2'].hex()}")
    print(f"Target key3: {real_tests['triple']['key3'].hex()}")
    print(f"Ciphertext: {real_tests['triple']['ciphertext'].hex()}")
    
    result = cracker.triple_des_bruteforce_real(
        real_tests['triple']['ciphertext'],
        real_tests['plaintext']
    )
    return result


def main():
    print("Welcome to DES Cracker - Cryptanalysis Tool By FuonDai")
    
    while True:
        show_menu()
        choice = input("Enter your choice (1-5): ").strip()
        
        try:
            if choice == '1':
                run_demo_mode()
                
            elif choice == '2':
                run_real_single_des()
                
            elif choice == '3':
                run_real_double_des()
                
            elif choice == '4':
                run_real_triple_des()
                
            elif choice == '5':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice! Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n\nOperation interrupted by user.")
            continue
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            continue
        
        print("\n" + "="*50)
        print("Attack completed. Returning to main menu...")


if __name__ == "__main__":
    main() 