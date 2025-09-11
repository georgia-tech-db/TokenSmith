Two-phase locking (2PL) is a protocol used in database systems to ensure **concurrency control** and **serializability** of transactions. Here's how it works:

### **1. Phases of 2PL**
- **Growing Phase**:  
  A transaction can **acquire locks** (shared or exclusive) on data items it needs to read/write. - **Shared locks (S)**: Allow other transactions to read but not write. - **Exclusive locks (X)**: Prevent other transactions from reading or writing. - **Shrinking Phase**:  
  A transaction can **release locks** but **cannot acquire new ones**.

### **2. Key Properties**
- **No Locking After Shrinking**: Once a transaction enters the shrinking phase, it cannot request new locks. - **Lock Points**: Transactions must **serialize** by their lock points (e.g., T1 acquires X on A, then T2 acquires X on A, ensuring order).

### **3. Example**  
- **T3 and T4**: Two-phase transactions (valid under 2PL). - **T1 and T2**: Not two-phase (violate 2PL rules).

### **4. Limitations**  
- **Blocking**: If a transaction tries to upgrade a lock (e.g., from S to X) and the item is locked by another transaction in S mode, it must **wait**. - **Blocking Problem**: In distributed systems, 2PL can cause **deadlocks** or **long waits** (e.g., in two-phase commit).

### **5. Variations**  
- **Lock Conversion**: Allows upgrading/downgrading locks (e.g., S → X). - **Multiversion 2PL**: Uses versioning to avoid blocking (used in commercial systems).

### **6. When to Use 2PL**  
- When **