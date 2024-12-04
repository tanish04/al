#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;
int comparisons = 0;
int partition(vector<int> &arr, int low, int high) {
    swap(arr[high], arr[low + rand() % (high - low + 1)]);
    int pivot = arr[high], i = low - 1;
    for (int j = low; j < high; j++) {
        comparisons++;
        if (arr[j] <= pivot) swap(arr[++i], arr[j]);
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quickSort(vector<int> &arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;
    vector<int> arr(n);
    cout << "Enter elements: ";
    for (int &x : arr) cin >> x;

    quickSort(arr, 0, n - 1);

    cout << "Sorted array: ";
    for (int x : arr) cout << x << " ";
    cout << "\nComparisons: " << comparisons << endl;
    return 0;
} 

//Random select

#include <iostream>
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time()

using namespace std;

// Function to swap two elements
void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

// Partition function with randomized pivot
int randomizedPartition(int arr[], int low, int high) {
    // Generate a random index for the pivot
    int randomIndex = low + rand() % (high - low + 1);
    swap(arr[randomIndex], arr[high]); // Move random pivot to the end

    int pivot = arr[high];
    int i = low - 1; // Index for smaller element

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1; // Return the pivot index
}

// Randomized Select function
int randomizedSelect(int arr[], int low, int high, int i) {
    if (low == high) {
        return arr[low]; // Only one element in the array
    }

    // Partition the array
    int pivotIndex = randomizedPartition(arr, low, high);

    // Number of elements in the left partition
    int k = pivotIndex - low + 1;

    if (i == k) {
        return arr[pivotIndex]; // The pivot is the i-th smallest element
    } else if (i < k) {
        return randomizedSelect(arr, low, pivotIndex - 1, i); // Search in the left partition
    } else {
        return randomizedSelect(arr, pivotIndex + 1, high, i - k); // Search in the right partition
    }
}

int main() {
    // Seed the random number generator
    srand(time(0));

    // Input array
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;

    int arr[n];
    cout << "Enter the elements of the array:\n";
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    int i;
    cout << "Enter the position of the smallest element to find (1-based): ";
    cin >> i;

    if (i < 1 || i > n) {
        cout << "Invalid position!" << endl;
        return 1;
    }

    // Find the i-th smallest element
    int result = randomizedSelect(arr, 0, n - 1, i);

    // Output the result
    cout << "The " << i << "-th smallest element is: " << result << endl;

    return 0;
}


//Kruskal 

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

struct Edge {
    int u, v, weight;
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

class Graph {
public:
    int V;
    vector<Edge> edges;

    Graph(int vertices) : V(vertices) {}

    void addEdge(int u, int v, int weight) {
        edges.push_back({u, v, weight});
    }

    int find(vector<int>& parent, int i) {
        if (parent[i] != i)
            parent[i] = find(parent, parent[i]);
        return parent[i];
    }

    void unite(vector<int>& parent, vector<int>& rank, int x, int y) {
        int xroot = find(parent, x);
        int yroot = find(parent, y);
        if (rank[xroot] < rank[yroot])
            parent[xroot] = yroot;
        else if (rank[xroot] > rank[yroot])
            parent[yroot] = xroot;
        else {
            parent[yroot] = xroot;
            rank[xroot]++;
        }
    }

    void kruskalMST() {
        sort(edges.begin(), edges.end());

        vector<int> parent(V);
        vector<int> rank(V, 0);
        for (int i = 0; i < V; ++i)
            parent[i] = i;

        vector<Edge> mst;
        for (const auto& edge : edges) {
            int u = find(parent, edge.u);
            int v = find(parent, edge.v);

            if (u != v) {
                mst.push_back(edge);
                unite(parent, rank, u, v);
            }
        }

        cout << "\nEdges in the Minimum Spanning Tree:\n";
        for (const auto& edge : mst)
            cout << edge.u << " - " << edge.v << " : " << edge.weight << "\n";
    }
};

int main() {
    int V, E;
    cout << "Enter the number of vertices: ";
    cin >> V;
    cout << "Enter the number of edges: ";
    cin >> E;

    Graph g(V);

    cout << "Enter the edges in the format (u v weight):\n";
    for (int i = 0; i < E; ++i) {
        int u, v, weight;
        cin >> u >> v >> weight;
        g.addEdge(u, v, weight);
    }

    g.kruskalMST();

    return 0;
}


//Bellman

#include <iostream>
#include <vector>
#include <climits> // For INT_MAX

using namespace std;

// Structure to represent an edge in the graph
struct Edge {
    int src, dest, weight;
};

// Bellman-Ford Algorithm
void bellmanFord(int V, int E, vector<Edge> &edges, int source) {
    // Initialize distances from the source to all vertices as infinite
    vector<int> distance(V, INT_MAX);
    distance[source] = 0;

    // Relax all edges V-1 times
    for (int i = 1; i <= V - 1; i++) {
        for (int j = 0; j < E; j++) {
            int u = edges[j].src;
            int v = edges[j].dest;
            int weight = edges[j].weight;

            if (distance[u] != INT_MAX && distance[u] + weight < distance[v]) {
                distance[v] = distance[u] + weight;
            }
        }
    }

    // Check for negative weight cycles
    for (int j = 0; j < E; j++) {
        int u = edges[j].src;
        int v = edges[j].dest;
        int weight = edges[j].weight;

        if (distance[u] != INT_MAX && distance[u] + weight < distance[v]) {
            cout << "Graph contains a negative weight cycle!" << endl;
            return;
        }
    }

    // Print the shortest distances
    cout << "Vertex\tDistance from Source" << endl;
    for (int i = 0; i < V; i++) {
        if (distance[i] == INT_MAX) {
            cout << i << "\tINFINITY" << endl;
        } else {
            cout << i << "\t" << distance[i] << endl;
        }
    }
}

int main() {
    int V, E, source;

    cout << "Enter the number of vertices and edges: ";
    cin >> V >> E;

    vector<Edge> edges(E);

    cout << "Enter the edges in the format (source destination weight):" << endl;
    for (int i = 0; i < E; i++) {
        cin >> edges[i].src >> edges[i].dest >> edges[i].weight;
    }

    cout << "Enter the source vertex: ";
    cin >> source;

    bellmanFord(V, E, edges, source);

    return 0;
}

//Btree

#include <iostream>
using namespace std;

struct BTreeNode {
    int *keys;       // Array of keys
    int t;           // Minimum degree
    BTreeNode **C;   // Array of child pointers
    int n;           // Current number of keys
    bool leaf;       // True if leaf node

    BTreeNode(int _t, bool _leaf) {
        t = _t;
        leaf = _leaf;
        keys = new int[2 * t - 1];
        C = new BTreeNode *[2 * t];
        n = 0;
    }

    void traverse() {
        for (int i = 0; i < n; i++) {
            if (!leaf) C[i]->traverse();
            cout << keys[i] << " ";
        }
        if (!leaf) C[n]->traverse();
    }

    BTreeNode* search(int k) {
        int i = 0;
        while (i < n && k > keys[i]) i++;
        if (i < n && keys[i] == k) return this;
        return leaf ? nullptr : C[i]->search(k);
    }

    void insertNonFull(int k) {
        int i = n - 1;
        if (leaf) {
            while (i >= 0 && keys[i] > k) {
                keys[i + 1] = keys[i];
                i--;
            }
            keys[i + 1] = k;
            n++;
        } else {
            while (i >= 0 && keys[i] > k) i--;
            if (C[i + 1]->n == 2 * t - 1) {
                splitChild(i + 1, C[i + 1]);
                if (keys[i + 1] < k) i++;
            }
            C[i + 1]->insertNonFull(k);
        }
    }

    void splitChild(int i, BTreeNode* y) {
        BTreeNode* z = new BTreeNode(y->t, y->leaf);
        z->n = t - 1;

        for (int j = 0; j < t - 1; j++) z->keys[j] = y->keys[j + t];
        if (!y->leaf) for (int j = 0; j < t; j++) z->C[j] = y->C[j + t];

        y->n = t - 1;

        for (int j = n; j >= i + 1; j--) C[j + 1] = C[j];
        C[i + 1] = z;

        for (int j = n - 1; j >= i; j--) keys[j + 1] = keys[j];
        keys[i] = y->keys[t - 1];
        n++;
    }
};

class BTree {
    BTreeNode* root;
    int t;

public:
    BTree(int _t) {
        root = nullptr;
        t = _t;
    }

    void traverse() {
        if (root) root->traverse();
        cout << endl;
    }

    BTreeNode* search(int k) {
        return root ? root->search(k) : nullptr;
    }

    void insert(int k) {
        if (!root) {
            root = new BTreeNode(t, true);
            root->keys[0] = k;
            root->n = 1;
        } else {
            if (root->n == 2 * t - 1) {
                BTreeNode* s = new BTreeNode(t, false);
                s->C[0] = root;
                s->splitChild(0, root);
                int i = (s->keys[0] < k) ? 1 : 0;
                s->C[i]->insertNonFull(k);
                root = s;
            } else {
                root->insertNonFull(k);
            }
        }
    }
};

int main() {
    int degree, choice, key;
    cout << "Enter the minimum degree of the B-Tree: ";
    cin >> degree;

    BTree t(degree);

    do {
        cout << "\nMenu:\n";
        cout << "1. Insert\n2. Search\n3. Traverse\n4. Exit\n";
        cout << "Enter your choice: ";
        cin >> choice;

        switch (choice) {
        case 1:
            cout << "Enter a key to insert: ";
            cin >> key;
            t.insert(key);
            break;
        case 2:
            cout << "Enter a key to search: ";
            cin >> key;
            cout << (t.search(key) ? "Key Found" : "Key Not Found") << endl;
            break;
        case 3:
            cout << "B-Tree keys: ";
            t.traverse();
            break;
        case 4:
            cout << "Exiting...\n";
            break;
        default:
            cout << "Invalid choice. Try again.\n";
        }
    } while (choice != 4);

    return 0;
}

//Treedatastructure

#include <iostream>
using namespace std;

// Node structure for the tree
struct TreeNode {
    int value;
    TreeNode* left;
    TreeNode* right;

    TreeNode(int val) : value(val), left(nullptr), right(nullptr) {}
};

// Binary Search Tree class
class BinaryTree {
private:
    TreeNode* root;

    // Helper function for insertion
    TreeNode* insert(TreeNode* node, int value) {
        if (!node) return new TreeNode(value); // Create a new node if empty
        if (value < node->value)
            node->left = insert(node->left, value); // Insert in the left subtree
        else if (value > node->value)
            node->right = insert(node->right, value); // Insert in the right subtree
        return node;
    }

    // Helper function for searching
    bool search(TreeNode* node, int value) {
        if (!node) return false;               // Value not found
        if (node->value == value) return true; // Value found
        if (value < node->value)
            return search(node->left, value);  // Search in the left subtree
        else
            return search(node->right, value); // Search in the right subtree
    }

    // Helper function for in-order traversal (optional, for visualization)
    void inorder(TreeNode* node) {
        if (!node) return;
        inorder(node->left);
        cout << node->value << " ";
        inorder(node->right);
    }

public:
    BinaryTree() : root(nullptr) {}

    // Insert a value
    void insert(int value) {
        root = insert(root, value);
    }

    // Search for a value
    bool search(int value) {
        return search(root, value);
    }

    // Display the tree (optional)
    void display() {
        inorder(root);
        cout << endl;
    }
};

int main() {
    BinaryTree tree;
    int choice, value;

    while (true) {
        cout << "1. Insert\n2. Search\n3. Display Tree\n4. Exit\nEnter your choice: ";
        cin >> choice;

        switch (choice) {
        case 1:
            cout << "Enter value to insert: ";
            cin >> value;
            tree.insert(value);
            break;
        case 2:
            cout << "Enter value to search: ";
            cin >> value;
            if (tree.search(value))
                cout << "Value found in the tree.\n";
            else
                cout << "Value not found in the tree.\n";
            break;
        case 3:
            cout << "Tree (in-order): ";
            tree.display();
            break;
        case 4:
            return 0;
        default:
            cout << "Invalid choice! Please try again.\n";
        }
    }
}

//KMP

#include <iostream>
#include <vector>
using namespace std;

// Function to compute the LPS (Longest Prefix Suffix) array
vector<int> computeLPS(string pattern) {
    int m = pattern.length();
    vector<int> lps(m, 0);
    int j = 0;

    for (int i = 1; i < m; i++) {
        while (j > 0 && pattern[i] != pattern[j]) {
            j = lps[j - 1];
        }
        if (pattern[i] == pattern[j]) {
            j++;
        }
        lps[i] = j;
    }
    return lps;
}

// KMP Search function
void kmpSearch(string text, string pattern) {
    vector<int> lps = computeLPS(pattern);
    int n = text.length(), m = pattern.length();
    int i = 0, j = 0;

    while (i < n) {
        if (text[i] == pattern[j]) {
            i++;
            j++;
        }
        if (j == m) {
            cout << "Pattern found at index " << i - j << endl;
            j = lps[j - 1];
        } else if (i < n && text[i] != pattern[j]) {
            if (j > 0) {
                j = lps[j - 1];
            } else {
                i++;
            }
        }
    }
}

int main() {
    string text, pattern;
    
    // Taking user input for text and pattern
    cout << "Enter the text: ";
    getline(cin, text); // Allows spaces in input
    cout << "Enter the pattern: ";
    getline(cin, pattern);

    // Perform KMP search
    kmpSearch(text, pattern);

    return 0;
}

//Suffix

#include <iostream>
#include <map>
#include <string>
#include <vector>
using namespace std;

// Node structure for the suffix tree
class SuffixTreeNode {
public:
    map<char, SuffixTreeNode*> children;
    int start; // Starting index of the edge label
    int* end;  // Ending index of the edge label
    int suffixIndex; // Suffix index for leaf nodes

    SuffixTreeNode(int start, int* end)
        : start(start), end(end), suffixIndex(-1) {}
};

class SuffixTree {
private:
    SuffixTreeNode* root;
    string text; // Input text
    int leafEnd; // End for all leaf nodes

    void buildSuffixTree();
    void extendSuffixTree(int pos);
    void setSuffixIndices(SuffixTreeNode* node, int height);
    void deleteTree(SuffixTreeNode* node);

    // Utility function to print the tree
    void printTree(SuffixTreeNode* node, int level = 0);

public:
    SuffixTree(const string& str) : text(str), root(nullptr), leafEnd(-1) {
        buildSuffixTree();
    }
    ~SuffixTree() { deleteTree(root); }
    void print() { printTree(root); }
};

// Build the suffix tree
void SuffixTree::buildSuffixTree() {
    root = new SuffixTreeNode(-1, new int(-1)); // Root node

    for (int i = 0; i < text.length(); i++) {
        extendSuffixTree(i);
    }

    setSuffixIndices(root, 0);
}

// Extend the suffix tree to include text[0...pos]
void SuffixTree::extendSuffixTree(int pos) {
    leafEnd = pos;

    static SuffixTreeNode* lastNewNode = nullptr;
    static SuffixTreeNode* activeNode = root;

    static int activeEdge = -1;
    static int activeLength = 0;

    static int remainingSuffixCount = 0;

    remainingSuffixCount++;

    while (remainingSuffixCount > 0) {
        if (activeLength == 0) activeEdge = pos;

        if (activeNode->children.find(text[activeEdge]) == activeNode->children.end()) {
            activeNode->children[text[activeEdge]] = new SuffixTreeNode(pos, &leafEnd);
            if (lastNewNode != nullptr) {
                lastNewNode->suffixIndex = activeNode->start;
                lastNewNode = nullptr;
            }
        } else {
            SuffixTreeNode* next = activeNode->children[text[activeEdge]];
            int edgeLength = *(next->end) - next->start + 1;

            if (activeLength >= edgeLength) {
                activeEdge += edgeLength;
                activeLength -= edgeLength;
                activeNode = next;
                continue;
            }

            if (text[next->start + activeLength] == text[pos]) {
                if (lastNewNode != nullptr && activeNode != root) {
                    lastNewNode->suffixIndex = activeNode->start;
                    lastNewNode = nullptr;
                }
                activeLength++;
                break;
            }

            int* splitEnd = new int(next->start + activeLength - 1);
            SuffixTreeNode* split = new SuffixTreeNode(next->start, splitEnd);
            activeNode->children[text[activeEdge]] = split;

            split->children[text[pos]] = new SuffixTreeNode(pos, &leafEnd);
            next->start += activeLength;
            split->children[text[next->start]] = next;

            if (lastNewNode != nullptr) {
                lastNewNode->suffixIndex = activeNode->start;
            }
            lastNewNode = split;
        }

        remainingSuffixCount--;

        if (activeNode == root && activeLength > 0) {
            activeLength--;
            activeEdge = pos - remainingSuffixCount + 1;
        } else if (activeNode != root) {
            activeNode = root;
        }
    }
}

// Set suffix indices for leaf nodes
void SuffixTree::setSuffixIndices(SuffixTreeNode* node, int height) {
    if (!node) return;

    if (node->children.empty()) {
        node->suffixIndex = text.length() - height;
        return;
    }

    for (auto& child : node->children) {
        setSuffixIndices(child.second, height + *(child.second->end) - child.second->start + 1);
    }
}

// Delete the suffix tree to free memory
void SuffixTree::deleteTree(SuffixTreeNode* node) {
    if (!node) return;

    for (auto& child : node->children) {
        deleteTree(child.second);
    }

    delete node->end;
    delete node;
}

// Print the suffix tree
void SuffixTree::printTree(SuffixTreeNode* node, int level) {
    if (!node) return;

    if (node->start != -1) {
        cout << string(level, '-') << text.substr(node->start, *(node->end) - node->start + 1) << endl;
    }

    for (auto& child : node->children) {
        printTree(child.second, level + 2);
    }
}

int main() {
    string text;
    cout << "Enter the text: ";
    cin >> text;

    SuffixTree tree(text + "$"); // Append '$' as a unique terminator
    tree.print();

    return 0;
}
