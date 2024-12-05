#include <iostream>
#include <vector>
#include <chrono>
#include <stack>
#include <numeric>
#include <algorithm>
#include <bitset>
#include <cstdlib>

#include "parser.hpp"

class BitString {
public:
    char* domains; 
    const size_t domains_size;
    const size_t N;

    char* singleton_domains;  // Array per tracciare i domini singleton
    char* checked_domains; // Array per tracciare i domini controllati

    size_t* remaining_value; // Array per tracciare i valori rimanenti
    size_t* canceled_values; // Array per tracciare il numero di valori cancellati

    // Costruttore
    BitString(const size_t& total_bytes, const size_t& N_)
    : domains(new char[total_bytes])
    , domains_size(total_bytes)
    , N(N_)
    , singleton_domains(new char[(N+7)/8])
    , checked_domains(new char[(N+7)/8])
    , remaining_value(new size_t[N])
    , canceled_values(new size_t[N]) 
    {                
        std::fill(singleton_domains, singleton_domains + (N+7)/8, 0x00);
        std::fill(domains, domains + total_bytes, 0x00);
        std::fill(checked_domains, checked_domains + (N+7)/8, 0x00);
        std::fill(remaining_value, remaining_value + N, 0);
        std::fill(canceled_values, canceled_values + N, 0);
    }

    // Distruttore
    ~BitString() 
    {
        delete[] domains; 
        delete[] singleton_domains;
        delete[] remaining_value;
        delete[] canceled_values;
        delete[] checked_domains;
    }

    // Costruttore di copia
    BitString(const BitString& other)
    : domains(new char[other.domains_size])
    , domains_size(other.domains_size)
    , N(other.N)
    , singleton_domains(new char[(other.N+7)/8])
    , checked_domains(new char[(other.N+7)/8])
    , remaining_value(new size_t[other.N])
    , canceled_values(new size_t[other.N]) 
    {
        std::copy(other.domains, other.domains + other.domains_size, domains);
        std::copy(other.singleton_domains, other.singleton_domains + (other.N+7)/8, singleton_domains);
        std::copy(other.checked_domains, other.checked_domains + (other.N+7)/8, checked_domains);
        std::copy(other.remaining_value, other.remaining_value + other.N, remaining_value);
        std::copy(other.canceled_values, other.canceled_values + other.N, canceled_values);
    }

    // Operatore di assegnazione
    BitString& operator=(const BitString& other) 
    {
        if (this != &other) {  // Evita auto-assegnazione
            delete[] domains; 
            delete[] singleton_domains;
            delete[] remaining_value;
            delete[] canceled_values;
            delete[] checked_domains;

            domains = new char[other.domains_size];
            singleton_domains = new char[(other.N+7)/8];
            checked_domains = new char[(other.N+7)/8];
            remaining_value = new size_t[other.N];
            canceled_values = new size_t[other.N];

            std::copy(other.domains, other.domains + other.domains_size, domains);
            std::copy(other.singleton_domains, other.singleton_domains + (other.N+7)/8, singleton_domains);
            std::copy(other.checked_domains, other.checked_domains + (other.N+7)/8, checked_domains);
            std::copy(other.remaining_value, other.remaining_value + other.N, remaining_value);
            std::copy(other.canceled_values, other.canceled_values + other.N, canceled_values);
        }
        return *this;
    }

    // Costruttore di move
    BitString(BitString&& other) noexcept
    : domains(other.domains)
    , domains_size(other.domains_size)
    , N(other.N)
    , singleton_domains(other.singleton_domains)
    , checked_domains(other.checked_domains)
    , remaining_value(other.remaining_value)
    , canceled_values(other.canceled_values)
    {
        other.domains = nullptr; 
        other.singleton_domains = nullptr;
        other.remaining_value = nullptr;
        other.canceled_values = nullptr;
        other.checked_domains = nullptr;
    }

    // Operatore di assegnazione di move
    BitString& operator=(BitString&& other) noexcept 
    {
        if (this != &other) {  // Evita auto-assegnazione
            delete[] domains; 
            delete[] singleton_domains;
            delete[] remaining_value;
            delete[] canceled_values;
            delete[] checked_domains;

            domains = other.domains;
            singleton_domains = other.singleton_domains;
            remaining_value = other.remaining_value;
            canceled_values = other.canceled_values;
            checked_domains = other.checked_domains;

            other.domains = nullptr;
            other.singleton_domains = nullptr;
            other.remaining_value = nullptr;
            other.canceled_values = nullptr;
            other.checked_domains = nullptr;
        }
        return *this;
    }

    bool is_domain_a_singleton(const size_t& domain_id)
    {
        const size_t byte_idx = domain_id / 8;
        const size_t bit_idx = domain_id % 8;

        return singleton_domains[byte_idx] & (1 << bit_idx);
    }

    void set_domain_singleton(const size_t& domain_id)
    {
        const size_t byte_idx = domain_id / 8;
        const size_t bit_idx = domain_id % 8;

        singleton_domains[byte_idx] |= (1 << bit_idx);
    }

    bool are_domains_singleton()
    {
        const size_t bytes = (N + 7) / 8;
        const size_t non_domain_bits = 8 - N % 8;
        // check if all bits are 1 apart from the last non_dom
        for (size_t curr_byte = 0; curr_byte < bytes - 1; ++curr_byte)
        {
            if (singleton_domains[curr_byte] != 0xFF)
            {
                return false;
            }
        }

        // check if the last byte is correct
        if (singleton_domains[bytes - 1] != (0xFF >> non_domain_bits))
        {
            return false;
        }
        return true;

    }

    void set_checked_domain(const size_t& domain_id)
    {
        const size_t byte_idx = domain_id / 8;
        const size_t bit_idx = domain_id % 8;

        checked_domains[byte_idx] |= (1 << bit_idx);
    }

    bool is_domain_checked(const size_t& domain_id)
    {
        const size_t byte_idx = domain_id / 8;
        const size_t bit_idx = domain_id % 8;

        return checked_domains[byte_idx] & (1 << bit_idx);
    }

    // Costruttore di default disabilitato
    BitString() = delete;

    // Metodo per annullare tutti gli elementi tranne uno
    void cancel_all_but_one_element_from_domain(const int& element, const size_t& domain_idx, const std::vector<size_t>& offsets, Data& d)
    {
        const size_t starting_byte_idx = offsets[domain_idx] / 8;
        const size_t starting_bit_idx = offsets[domain_idx] % 8;

        for (size_t current_bit = 0; current_bit <= d.get_u_at(domain_idx); ++current_bit)
        {
            const size_t byte_idx = starting_byte_idx + (starting_bit_idx + current_bit) / 8;
            const size_t bit_idx = (starting_bit_idx + current_bit) % 8;

            if (current_bit != element)
            {
                domains[byte_idx] |= (1 << bit_idx);
            }
        }
    }
};

void fixpoint(BitString& current, std::stack<BitString>& stack, const size_t& N, const std::vector<size_t>& offsets, Data& d, size_t& exploredTree, size_t& exploredSol)
{
    bool changed = true;

    while (changed)
    {
        changed = false;

        for (size_t domain_idx = 0; domain_idx < N; ++domain_idx)
        {   
            if (!current.is_domain_a_singleton(domain_idx) || current.is_domain_checked(domain_idx)) continue;

            const int value = current.remaining_value[domain_idx];

            for (size_t other_variable = 0; other_variable < N; ++other_variable)
            {
                if (other_variable == domain_idx) continue;
                if (d.get_C_at(domain_idx, other_variable) == 1)
                {
                    if (value <= d.get_u_at(other_variable))
                    {
                        const size_t byte_idx = (offsets[other_variable] + value) / 8;
                        const size_t bit_idx = (offsets[other_variable] + value) % 8;

                        if (current.domains[byte_idx] & (1 << bit_idx)) continue;

                        current.domains[byte_idx] |= (1 << bit_idx);
                        current.canceled_values[other_variable]++;

                        if (current.canceled_values[other_variable] == d.get_u_at(other_variable) + 1)
                            return;

                        if (current.canceled_values[other_variable] == d.get_u_at(other_variable))
                        {
                            current.set_domain_singleton(other_variable);
                            changed = true;

                            for (size_t possible_remaining = 0; possible_remaining <= d.get_u_at(other_variable); ++possible_remaining)
                            {
                                const size_t byte_idx = (offsets[other_variable] + possible_remaining) / 8;
                                const size_t bit_idx = (offsets[other_variable] + possible_remaining) % 8;
                                if (!(current.domains[byte_idx] & (1 << (bit_idx))))
                                {
                                    current.remaining_value[other_variable] = possible_remaining;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            current.set_checked_domain(domain_idx);
        }
    }

    if (current.are_domains_singleton())
    {
        exploredSol++;
        return;
    }

    size_t branching_variable = 0;
    for (size_t i = 0; i < N; ++i)
    {
        if (!current.is_domain_a_singleton(i))
        {
            branching_variable = i;
            break;
        }
    }

    // If the branching variable is the last one, we just add to the solutions the number 
    // of remaining values
    if (branching_variable == N - 1)
    {
        exploredSol += d.get_u_at(N - 1) - current.canceled_values[N - 1] + 1;
        return;
    }

    for (size_t branch_value = 0; branch_value <= d.get_u_at(branching_variable); ++branch_value)
    {
        const size_t byte_idx = (offsets[branching_variable] + branch_value) / 8;
        const size_t bit_idx = (offsets[branching_variable] + branch_value) % 8;

        if (current.domains[byte_idx] & (1 << bit_idx)) continue;

        BitString child(current);
        child.set_domain_singleton(branching_variable);
        child.remaining_value[branching_variable] = branch_value;

        child.cancel_all_but_one_element_from_domain(branch_value, branching_variable, offsets, d);

        child.canceled_values[branching_variable] = d.get_u_at(branching_variable);

        stack.push(std::move(child));
        exploredTree++;
    }
}

void test_fixpoint(char* str)
{
    Data d;
    d.read_input(str);

    size_t total_bytes = 0;
    size_t total_values = 0;

    std::vector<size_t> offsets(d.get_n(), 0);
    offsets[0] = 0;
    for (size_t i = 1; i < d.get_n(); ++i)
    {
        offsets[i] = offsets[i-1] + d.get_u_at(i-1) + 1;
    }

    for (size_t i = 0; i < d.get_n(); ++i)
    {
        total_values += d.get_u_at(i) + 1;
    }

    total_bytes = (total_values + 7) / 8;

    BitString root(total_bytes, d.get_n());
    std::stack<BitString> stack;
    stack.push(std::move(root));

    size_t exploredTree = 0, exploredSol = 0;

    auto start = std::chrono::high_resolution_clock::now();
    while (!stack.empty())
    {
        BitString current = std::move(stack.top());
        stack.pop();

        fixpoint(current, stack, d.get_n(), offsets, d, exploredTree, exploredSol);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Explored tree : " << exploredTree << std::endl;
    std::cout << "Explored sol  : " << exploredSol << std::endl;
    std::cout << "Time          : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
}

int main(int argc, char** argv)
{
    test_fixpoint(argv[1]);
    return 0;
}