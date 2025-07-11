# Complete Go Cheat Sheet & Pro Practices

## Table of Contents

1. [Basic Syntax](#basic-syntax)
2. [Data Types & Variables](#data-types--variables)
3. [Control Structures](#control-structures)
4. [Functions](#functions)
5. [Structs & Interfaces](#structs--interfaces)
6. [Goroutines & Concurrency](#goroutines--concurrency)
7. [Error Handling](#error-handling)
8. [Packages & Modules](#packages--modules)
9. [HTTP Server Development](#http-server-development)
10. [Database Operations](#database-operations)
11. [Testing](#testing)
12. [Pro Tips & Best Practices](#pro-tips--best-practices)

---

## Basic Syntax

### Hello World

```go
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
```

### Variable Declaration

```go
// Multiple ways to declare variables
var name string = "John"
var age int = 25
var isActive bool = true

// Short declaration (inside functions only)
name := "John"
age := 25

// Multiple variables
var a, b, c int = 1, 2, 3
x, y, z := 1, 2, 3
```

### Constants

```go
const Pi = 3.14159
const (
    StatusOK = 200
    StatusNotFound = 404
    StatusError = 500
)

// iota for auto-incrementing
const (
    Sunday = iota    // 0
    Monday          // 1
    Tuesday         // 2
)
```

---

## Data Types & Variables

### Basic Types

```go
// Numbers
var i int = 42
var f float64 = 3.14
var c complex64 = 1 + 2i

// Strings
var s string = "Hello"
var r rune = 'A'  // Unicode code point

// Boolean
var b bool = true
```

### Collections

```go
// Arrays (fixed size)
var arr [5]int = [5]int{1, 2, 3, 4, 5}
arr := [...]int{1, 2, 3, 4, 5}  // Auto-size

// Slices (dynamic arrays)
var slice []int = []int{1, 2, 3}
slice = append(slice, 4, 5)
slice = slice[1:3]  // Slicing

// Maps
var m map[string]int = make(map[string]int)
m["key"] = 42
delete(m, "key")

// Map literal
m := map[string]int{
    "apple":  5,
    "banana": 3,
}
```

### Pointers

```go
var p *int
i := 42
p = &i
fmt.Println(*p)  // Dereference pointer

// Creating pointer with new
p := new(int)
*p = 42
```

---

## Control Structures

### Conditionals

```go
if x > 0 {
    fmt.Println("Positive")
} else if x < 0 {
    fmt.Println("Negative")
} else {
    fmt.Println("Zero")
}

// If with statement
if err := doSomething(); err != nil {
    log.Fatal(err)
}

// Switch
switch day {
case "Monday":
    fmt.Println("Start of work week")
case "Friday":
    fmt.Println("TGIF")
default:
    fmt.Println("Regular day")
}

// Switch without expression
switch {
case x > 0:
    fmt.Println("Positive")
case x < 0:
    fmt.Println("Negative")
}
```

### Loops

```go
// For loop
for i := 0; i < 10; i++ {
    fmt.Println(i)
}

// While-like loop
for condition {
    // code
}

// Infinite loop
for {
    break  // Exit condition
}

// Range over slice/array
for i, value := range slice {
    fmt.Printf("Index: %d, Value: %d\n", i, value)
}

// Range over map
for key, value := range m {
    fmt.Printf("Key: %s, Value: %d\n", key, value)
}
```

---

## Functions

### Basic Functions

```go
func add(a, b int) int {
    return a + b
}

// Multiple return values
func divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}

// Named return values
func swap(x, y string) (a, b string) {
    a = y
    b = x
    return  // naked return
}
```

### Advanced Function Features

```go
// Variadic functions
func sum(numbers ...int) int {
    total := 0
    for _, num := range numbers {
        total += num
    }
    return total
}

// Function as type
type Calculator func(int, int) int

func compute(fn Calculator, a, b int) int {
    return fn(a, b)
}

// Closures
func counter() func() int {
    count := 0
    return func() int {
        count++
        return count
    }
}
```

---

## Structs & Interfaces

### Structs

```go
type Person struct {
    Name string
    Age  int
}

// Constructor pattern
func NewPerson(name string, age int) *Person {
    return &Person{
        Name: name,
        Age:  age,
    }
}

// Methods
func (p *Person) Greet() string {
    return fmt.Sprintf("Hello, I'm %s", p.Name)
}

// Embedded structs
type Employee struct {
    Person
    ID       int
    Position string
}
```

### Interfaces

```go
type Writer interface {
    Write([]byte) (int, error)
}

type Stringer interface {
    String() string
}

// Interface implementation
func (p Person) String() string {
    return fmt.Sprintf("%s (%d years old)", p.Name, p.Age)
}

// Empty interface
func printAnything(v interface{}) {
    fmt.Println(v)
}

// Type assertion
if s, ok := v.(string); ok {
    fmt.Println("String:", s)
}

// Type switch
switch v := v.(type) {
case string:
    fmt.Println("String:", v)
case int:
    fmt.Println("Integer:", v)
default:
    fmt.Println("Unknown type")
}
```

---

## Goroutines & Concurrency

### Goroutines

```go
// Start a goroutine
go func() {
    fmt.Println("Running in goroutine")
}()

// Wait for goroutines
var wg sync.WaitGroup
wg.Add(2)

go func() {
    defer wg.Done()
    // work
}()

go func() {
    defer wg.Done()
    // work
}()

wg.Wait()
```

### Channels

```go
// Create channel
ch := make(chan int)
buffered := make(chan int, 10)

// Send and receive
go func() {
    ch <- 42
}()
value := <-ch

// Channel directions
func sender(ch chan<- int) {
    ch <- 42
}

func receiver(ch <-chan int) {
    value := <-ch
    fmt.Println(value)
}

// Select statement
select {
case msg1 := <-ch1:
    fmt.Println("Received from ch1:", msg1)
case msg2 := <-ch2:
    fmt.Println("Received from ch2:", msg2)
case <-time.After(1 * time.Second):
    fmt.Println("Timeout")
default:
    fmt.Println("No communication")
}
```

### Sync Primitives

```go
// Mutex
var mu sync.Mutex
var counter int

func increment() {
    mu.Lock()
    defer mu.Unlock()
    counter++
}

// RWMutex
var rwmu sync.RWMutex
var data map[string]int

func readData(key string) int {
    rwmu.RLock()
    defer rwmu.RUnlock()
    return data[key]
}

// Once
var once sync.Once

func initialize() {
    once.Do(func() {
        fmt.Println("Initialized once")
    })
}
```

---

## Error Handling

### Basic Error Handling

```go
func divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}

// Check errors
result, err := divide(10, 0)
if err != nil {
    log.Fatal(err)
}
```

### Custom Errors

```go
type ValidationError struct {
    Field string
    Value string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("invalid %s: %s", e.Field, e.Value)
}

// Error wrapping (Go 1.13+)
import "fmt"

func processFile(filename string) error {
    err := openFile(filename)
    if err != nil {
        return fmt.Errorf("failed to process %s: %w", filename, err)
    }
    return nil
}

// Error unwrapping
if errors.Is(err, ErrNotFound) {
    // handle not found
}

var validationErr *ValidationError
if errors.As(err, &validationErr) {
    // handle validation error
}
```

---

## Packages & Modules

### Package Structure

```go
// main.go
package main

import (
    "fmt"
    "myproject/internal/user"
    "github.com/gin-gonic/gin"
)

// user/user.go
package user

type User struct {
    ID   int
    Name string
}

func NewUser(name string) *User {
    return &User{Name: name}
}
```

### Go Modules

```bash
# Initialize module
go mod init myproject

# Add dependency
go get github.com/gin-gonic/gin

# Update dependencies
go mod tidy

# Vendor dependencies
go mod vendor
```

---

## HTTP Server Development

### Basic HTTP Server

```go
package main

import (
    "encoding/json"
    "log"
    "net/http"
)

type User struct {
    ID   int    `json:"id"`
    Name string `json:"name"`
}

func main() {
    http.HandleFunc("/users", usersHandler)
    http.HandleFunc("/users/", userHandler)

    log.Println("Server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}

func usersHandler(w http.ResponseWriter, r *http.Request) {
    switch r.Method {
    case http.MethodGet:
        getUsers(w, r)
    case http.MethodPost:
        createUser(w, r)
    default:
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
    }
}
```

### Advanced HTTP Server with Middleware

```go
package main

import (
    "context"
    "log"
    "net/http"
    "strings"
    "time"
)

// Middleware type
type Middleware func(http.Handler) http.Handler

// Logger middleware
func Logger(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        next.ServeHTTP(w, r)
        log.Printf("%s %s %v", r.Method, r.URL.Path, time.Since(start))
    })
}

// Auth middleware
func Auth(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        token := r.Header.Get("Authorization")
        if token == "" {
            http.Error(w, "Unauthorized", http.StatusUnauthorized)
            return
        }

        // Add user to context
        ctx := context.WithValue(r.Context(), "user", getUserFromToken(token))
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}

// Chain middlewares
func Chain(middlewares ...Middleware) Middleware {
    return func(next http.Handler) http.Handler {
        for i := len(middlewares) - 1; i >= 0; i-- {
            next = middlewares[i](next)
        }
        return next
    }
}

// Usage
func main() {
    mux := http.NewServeMux()
    mux.HandleFunc("/api/users", usersHandler)

    handler := Chain(Logger, Auth)(mux)

    server := &http.Server{
        Addr:         ":8080",
        Handler:      handler,
        ReadTimeout:  15 * time.Second,
        WriteTimeout: 15 * time.Second,
    }

    log.Fatal(server.ListenAndServe())
}
```

### JSON API with Validation

```go
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    "strconv"
    "strings"
)

type CreateUserRequest struct {
    Name  string `json:"name" validate:"required"`
    Email string `json:"email" validate:"required,email"`
}

type User struct {
    ID    int    `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}

func createUser(w http.ResponseWriter, r *http.Request) {
    var req CreateUserRequest

    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }

    // Validate request
    if err := validateCreateUserRequest(req); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    // Create user
    user := User{
        ID:    generateID(),
        Name:  req.Name,
        Email: req.Email,
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}

func validateCreateUserRequest(req CreateUserRequest) error {
    if req.Name == "" {
        return fmt.Errorf("name is required")
    }
    if req.Email == "" {
        return fmt.Errorf("email is required")
    }
    if !strings.Contains(req.Email, "@") {
        return fmt.Errorf("invalid email format")
    }
    return nil
}
```

---

## Database Operations

### SQL Database (PostgreSQL)

```go
package main

import (
    "database/sql"
    "log"

    _ "github.com/lib/pq"
)

type User struct {
    ID    int
    Name  string
    Email string
}

type UserRepo struct {
    db *sql.DB
}

func NewUserRepo(db *sql.DB) *UserRepo {
    return &UserRepo{db: db}
}

func (r *UserRepo) Create(user *User) error {
    query := `
        INSERT INTO users (name, email)
        VALUES ($1, $2)
        RETURNING id`

    err := r.db.QueryRow(query, user.Name, user.Email).Scan(&user.ID)
    return err
}

func (r *UserRepo) GetByID(id int) (*User, error) {
    user := &User{}
    query := `SELECT id, name, email FROM users WHERE id = $1`

    err := r.db.QueryRow(query, id).Scan(&user.ID, &user.Name, &user.Email)
    if err != nil {
        return nil, err
    }
    return user, nil
}

func (r *UserRepo) List() ([]*User, error) {
    query := `SELECT id, name, email FROM users`
    rows, err := r.db.Query(query)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    var users []*User
    for rows.Next() {
        user := &User{}
        err := rows.Scan(&user.ID, &user.Name, &user.Email)
        if err != nil {
            return nil, err
        }
        users = append(users, user)
    }
    return users, nil
}

// Connection setup
func setupDB() (*sql.DB, error) {
    db, err := sql.Open("postgres", "postgres://user:password@localhost/mydb?sslmode=disable")
    if err != nil {
        return nil, err
    }

    db.SetMaxOpenConns(10)
    db.SetMaxIdleConns(5)

    return db, nil
}
```

### Context and Transactions

```go
func (r *UserRepo) CreateWithContext(ctx context.Context, user *User) error {
    query := `INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id`
    err := r.db.QueryRowContext(ctx, query, user.Name, user.Email).Scan(&user.ID)
    return err
}

func (r *UserRepo) UpdateUserAndLog(ctx context.Context, user *User) error {
    tx, err := r.db.BeginTx(ctx, nil)
    if err != nil {
        return err
    }
    defer tx.Rollback()

    // Update user
    _, err = tx.ExecContext(ctx, "UPDATE users SET name=$1, email=$2 WHERE id=$3",
        user.Name, user.Email, user.ID)
    if err != nil {
        return err
    }

    // Log the update
    _, err = tx.ExecContext(ctx, "INSERT INTO user_logs (user_id, action) VALUES ($1, $2)",
        user.ID, "updated")
    if err != nil {
        return err
    }

    return tx.Commit()
}
```

---

## Testing

### Unit Tests

```go
package main

import (
    "testing"
    "reflect"
)

func TestAdd(t *testing.T) {
    result := add(2, 3)
    expected := 5

    if result != expected {
        t.Errorf("add(2, 3) = %d; want %d", result, expected)
    }
}

func TestAddTable(t *testing.T) {
    tests := []struct {
        name     string
        a, b     int
        expected int
    }{
        {"positive numbers", 2, 3, 5},
        {"negative numbers", -2, -3, -5},
        {"zero", 0, 5, 5},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := add(tt.a, tt.b)
            if result != tt.expected {
                t.Errorf("add(%d, %d) = %d; want %d", tt.a, tt.b, result, tt.expected)
            }
        })
    }
}
```

### HTTP Testing

```go
package main

import (
    "net/http"
    "net/http/httptest"
    "strings"
    "testing"
)

func TestUsersHandler(t *testing.T) {
    req := httptest.NewRequest(http.MethodGet, "/users", nil)
    w := httptest.NewRecorder()

    usersHandler(w, req)

    if w.Code != http.StatusOK {
        t.Errorf("Expected status %d, got %d", http.StatusOK, w.Code)
    }

    expected := `[{"id":1,"name":"John"}]`
    if strings.TrimSpace(w.Body.String()) != expected {
        t.Errorf("Expected body %s, got %s", expected, w.Body.String())
    }
}
```

### Benchmarks

```go
func BenchmarkAdd(b *testing.B) {
    for i := 0; i < b.N; i++ {
        add(2, 3)
    }
}

func BenchmarkStringBuilder(b *testing.B) {
    for i := 0; i < b.N; i++ {
        var builder strings.Builder
        for j := 0; j < 1000; j++ {
            builder.WriteString("test")
        }
        _ = builder.String()
    }
}
```

---

## Pro Tips & Best Practices

### Code Organization

```go
// Use clear package structure
myproject/
├── cmd/
│   └── server/
│       └── main.go
├── internal/
│   ├── user/
│   │   ├── handler.go
│   │   ├── service.go
│   │   └── repository.go
│   └── config/
│       └── config.go
├── pkg/
│   └── middleware/
│       └── auth.go
├── go.mod
└── go.sum
```

### Error Handling Best Practices

```go
// Don't ignore errors
result, err := doSomething()
if err != nil {
    return fmt.Errorf("failed to do something: %w", err)
}

// Use sentinel errors
var ErrNotFound = errors.New("not found")

// Custom error types
type ValidationError struct {
    Field string
    Value interface{}
}

func (e ValidationError) Error() string {
    return fmt.Sprintf("validation failed for field %s: %v", e.Field, e.Value)
}
```

### Performance Tips

```go
// Use string builder for concatenation
var builder strings.Builder
for _, s := range strings {
    builder.WriteString(s)
}
result := builder.String()

// Pre-allocate slices when size is known
items := make([]Item, 0, expectedSize)

// Use buffered channels for producer-consumer
ch := make(chan Item, 100)

// Pool expensive objects
var pool = sync.Pool{
    New: func() interface{} {
        return &ExpensiveObject{}
    },
}

obj := pool.Get().(*ExpensiveObject)
defer pool.Put(obj)
```

### Configuration Management

```go
package config

import (
    "os"
    "strconv"
)

type Config struct {
    Port        int
    DatabaseURL string
    JWTSecret   string
}

func Load() (*Config, error) {
    port, err := strconv.Atoi(getEnv("PORT", "8080"))
    if err != nil {
        return nil, err
    }

    return &Config{
        Port:        port,
        DatabaseURL: getEnv("DATABASE_URL", ""),
        JWTSecret:   getEnv("JWT_SECRET", ""),
    }, nil
}

func getEnv(key, defaultValue string) string {
    if value := os.Getenv(key); value != "" {
        return value
    }
    return defaultValue
}
```

### Graceful Shutdown

```go
package main

import (
    "context"
    "log"
    "net/http"
    "os"
    "os/signal"
    "syscall"
    "time"
)

func main() {
    server := &http.Server{
        Addr:    ":8080",
        Handler: setupRoutes(),
    }

    // Start server in goroutine
    go func() {
        if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
            log.Fatal("Server failed to start:", err)
        }
    }()

    // Wait for interrupt signal
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    <-quit

    log.Println("Shutting down server...")

    // Graceful shutdown with timeout
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    if err := server.Shutdown(ctx); err != nil {
        log.Fatal("Server forced to shutdown:", err)
    }

    log.Println("Server exited")
}
```

### Common Patterns

```go
// Option pattern
type Server struct {
    port int
    host string
}

type Option func(*Server)

func WithPort(port int) Option {
    return func(s *Server) {
        s.port = port
    }
}

func WithHost(host string) Option {
    return func(s *Server) {
        s.host = host
    }
}

func NewServer(opts ...Option) *Server {
    s := &Server{
        port: 8080,
        host: "localhost",
    }

    for _, opt := range opts {
        opt(s)
    }

    return s
}

// Usage
server := NewServer(
    WithPort(9000),
    WithHost("0.0.0.0"),
)
```

### Key Takeaways

1. **Always handle errors** - Don't ignore them
2. **Use interfaces** - Make your code testable and flexible
3. **Prefer composition over inheritance** - Embed structs when needed
4. **Use context** - For cancellation and request-scoped values
5. **Write tests** - Unit tests and integration tests
6. **Profile your code** - Use `go tool pprof` for performance analysis
7. **Keep packages small** - Single responsibility principle
8. **Use golint and go vet** - Maintain code quality
9. **Document public APIs** - Write clear godoc comments
10. **Think about concurrency** - Use goroutines and channels effectively

---

This cheat sheet covers the essential Go concepts and patterns you'll need for both general programming and server development. Remember to always refer to the official Go documentation for the most up-to-date information.
