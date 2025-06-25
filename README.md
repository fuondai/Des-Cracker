# DES Cracker - Educational Cryptanalysis Tool

## Mô tả

Đây là một công cụ giáo dục để minh họa các cuộc tấn công brute-force và meet-in-the-middle đối với thuật toán mã hóa DES, Double DES và Triple DES. Chương trình được thiết kế cho mục đích học tập và nghiên cứu, sử dụng không gian khóa giảm để demonstration có thể hoàn thành trong thời gian hợp lý.

## Tính năng

- **Single DES Brute-force Attack**: Thử tất cả khóa có thể để crack DES
- **Double DES Meet-in-the-Middle Attack**: Tấn công hiệu quả với complexity giảm
- **Test Vector Generation**: Tạo ciphertext để demo attacks
- **Performance Analysis**: Đo đạc thời gian và số lần thử

## Yêu cầu hệ thống

- Python 3.10 hoặc cao hơn
- PyCryptodome library

## Cài đặt

1. Clone hoặc download repository này
2. Cài đặt dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Sử dụng

### Chạy demo đầy đủ:

```bash
py -3.10 des_cracker.py
```

### Import như module:

```python
from des_cracker import DESCracker, TestVectorGenerator

# Tạo cracker instance
cracker = DESCracker()

# Tạo test vector
test = TestVectorGenerator.generate_single_des_test()

# Thực hiện attack
result = cracker.single_des_bruteforce(
    test['ciphertext'],
    test['plaintext'][:4]  # partial known plaintext
)
```

## Cấu trúc code

### DESCracker Class

**Main Methods:**

- `single_des_bruteforce()`: Brute-force attack trên single DES
- `double_des_meet_in_middle()`: Meet-in-the-middle attack trên Double DES
- `triple_des_bruteforce()`: Limited brute-force trên Triple DES

**Helper Methods:**

- `_generate_reduced_keys()`: Generate reduced keyspace cho demo
- `_pad_key_to_des_length()`: Pad short keys to DES length
- `_is_valid_plaintext()`: Validate decrypted plaintext

### TestVectorGenerator Class

Tạo test vectors với known plaintexts và keys để demonstrate attacks:

- `generate_single_des_test()`
- `generate_double_des_test()`
- `generate_triple_des_test()`

## Giải thích thuật toán

### 1. Single DES Brute-force

```
FOR mỗi possible_key IN keyspace:
    decrypted = DES_decrypt(ciphertext, possible_key)
    IF is_valid_plaintext(decrypted):
        RETURN possible_key, decrypted
```

**Complexity:** O(2^n) với n = key length in bits

### 2. Double DES Meet-in-the-Middle

**Phase 1 - Build encryption table:**

```
FOR mỗi K1 IN all_possible_K1:
    intermediate = E_K1(known_plaintext)
    encryption_table[intermediate] = K1
```

**Phase 2 - Search for matches:**

```
FOR mỗi K2 IN all_possible_K2:
    intermediate = D_K2(ciphertext)
    IF intermediate IN encryption_table:
        K1 = encryption_table[intermediate]
        RETURN K1, K2
```

**Complexity:** O(2^(n+1)) time, O(2^n) space instead of O(2^(2n))

### 3. Key Space Reduction

Để demonstration hoàn thành trong thời gian hợp lý:

- Single DES: 1-3 bytes thay vì 8 bytes
- Double DES: 1-2 bytes per key
- Triple DES: Limited attempts (10,000)

## Ví dụ output

```
DES CRYPTANALYSIS DEMONSTRATION
==================================================

1. SINGLE DES ATTACK DEMONSTRATION
----------------------------------------
Test vector: Single DES with reduced key (first 2 bytes: 01 02)
Original plaintext: b'HELLO123'
Original key: 0102000000000000
Ciphertext: a1b2c3d4e5f67890

Starting Single DES brute-force attack...
Trying keys of length 1...
Trying keys of length 2...

*** SUCCESS! ***
Key found: 0102 (padded: 0102000000000000)
Attempts: 258
Time: 0.05 seconds
Decrypted: b'HELLO123'
```

## Giới hạn và cảnh báo

### Educational Limitations

1. **Reduced Keyspace**: Chỉ sử dụng 2-3 byte đầu của key
2. **Simplified Validation**: Chỉ check ASCII characters
3. **Known Plaintext**: Demo scenario với known plaintext
4. **Limited Scope**: Không cover all real-world attack vectors

### Real-world Reality

- **Single DES**: Hoàn toàn không an toàn, đã bị crack trong vài giờ
- **Double DES**: Vulnerable to meet-in-the-middle với adequate resources
- **Triple DES**: Vẫn secure nhưng đã deprecated, thay thế bằng AES
- **Full keyspace**: 2^56 keys cho DES = 72,057,594,037,927,936 keys

## Complexity Analysis

| Attack Type        | Keyspace | Time Complexity | Space Complexity |
| ------------------ | -------- | --------------- | ---------------- |
| Single DES         | 2^56     | O(2^55) avg     | O(1)             |
| Double DES (naive) | 2^112    | O(2^111) avg    | O(1)             |
| Double DES (MITM)  | 2^112    | O(2^57)         | O(2^56)          |
| Triple DES         | 2^168    | O(2^167) avg    | O(1)             |

## Security Notes

⚠️ **CẢNH BÁO:** Tool này chỉ dành cho mục đích educational và research.

## Tài liệu tham khảo

- [NIST DES Specification](https://csrc.nist.gov/publications/detail/fips/46-3/archive/1999-10-25)
- [Meet-in-the-Middle Attack Paper](https://en.wikipedia.org/wiki/Meet-in-the-middle_attack)
