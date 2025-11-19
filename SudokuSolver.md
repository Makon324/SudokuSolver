# Rozwiązywacz Sudoku z Akceleracją GPU dla Wielu Plansz

Ten dokument opisuje podejście do rozwiązywania wielu łamigłówek Sudoku równolegle przy użyciu CUDA na GPU. Metoda wykorzystuje algorytm z nawrotami podobny do BFS. Obsługuje partie plansz, rozwijając drzewo wyszukiwania poziom po poziomie do maksymalnej głębokości 81 (liczba komórek w siatce Sudoku).

## Reprezentacja Planszy

Każda plansza Sudoku jest reprezentowana kompaktowo za pomocą tablicy masek bitowych składającej się z 11 elementów `uint32_t`, co daje łącznie 352 bity (potrzebne tylko 4 * 9 * 9 = 324). Ta struktura dzieli się na cztery logiczne sekcje, każda o 81 bitach (9x9), aby śledzić ograniczenia i stan:

- **Wiersze (bity 0–80)**: Dla każdego wiersza (0–8), 9 bitów wskazuje, które liczby (1–9) są użyte.
- **Kolumny (bity 81–161)**: Podobnie, dla każdej kolumny, bity śledzą zajęte liczby.
- **Bloki (bity 162–242)**: Dla każdego podsiatki 3x3 (9 łącznie), bity oznaczają użyte liczby.
- **Presetowane (bity 243–323)**: Bit na komórkę (81 łącznie) flaguje, czy jest ona wypełniona.

Ten format pakowany bitowo umożliwia szybkie sprawdzanie ograniczeń za pomocą operacji bitowych (np. OR do ustawienia, AND do usunięcia oraz sprawdzanie konfliktów). Jest efektywny pamięciowo. Wypełnienie do 11 `uint32_t` zapewnia wyrównanie, z ignorowaniem nieużywanych bitów.

## Struktura Tablic (SoA)

Aby zoptymalizować pod kątem równoległości GPU i koalescencji pamięci, plansze są przechowywane w układzie Struktury Tablic (SoA) za pomocą struktury `SudokuBoards`:

```cpp
struct SudokuBoards {
    int ids[];           // Tablica identyfikatorów plansz do śledzenia (1 łamigłówka 0; 2 - 1; ...)
    uint32_t repr[11][]; // Tablica 2D: 11 wierszy (segmenty masek bitowych) x N kolumn (plansz)
};
```

W przeciwieństwie do Tablicy Struktur (AoS), gdzie dane każdej planszy są ciągłe, SoA grupuje podobne dane razem (np. wszystkie maski bitowe wierszy dla plansz). To promuje koalescencyjne dostępy do pamięci w kernelach, ponieważ wątki przetwarzające różne plansze czytają/zapisują sąsiednie elementy. Dla N plansz, `repr[11][N]` pozwala na efektywność na poziomie warp podczas operacji pamięciowych.

## Jak Działa Podejście

Rozwiązywacz używa równoległego algorytmu przeszukiwania wstecznego do eksploracji przestrzeni wyszukiwania dla wielu plansz jednocześnie. Przetwarza plansze w partiach, iterując przez do `MAX_LEVELS = 81` poziomów. Na każdym poziomie:

1. **Kernel Znajdowania Następnej Komórki (`find_next_cell_kernel`)**: Uruchamiany na GPU, ten kernel skanuje każdą planszę, aby wybrać następną pustą komórkę za pomocą heurystyki Minimum Remaining Values (MRV) — wybierając komórkę z najmniejszą liczbą możliwych liczb, aby zminimalizować rozgałęzianie. Wyprowadza współrzędne (`next_cells_x`, `next_cells_y`) na planszę. Rozwiązane plansze są oznaczane jako (na razie) (200, 200), nieważne jako (255, 255).

2. **Kernel Generowania Dzieci (`generate_children_kernel`)**: Dla każdej planszy, próbuje ważnych liczb (1–9) w wybranej komórce, tworząc plansze potomne poprzez kopiowanie rodzica i aktualizację masek bitowych. Wyjścia są zapisywane do nowej partii `SudokuBoards`, z `out_inserted` śledzącym nowe plansze na wejście. To rozwija drzewo równolegle, wykorzystując wątki GPU dla wysokiej przepustowości.

Funkcja hosta `solve_multiple_sudoku` orkiestruje to:

- Alokuje pamięć urządzenia dla wejść/wyjść.
- Pętla po poziomach: Uruchamia kernel znajdowania, zbiera rozwiązane/nieważne plansze, następnie uruchamia kernel generowania, aby wyprodukować następną partię.
- Śledzi aktywne plansze za pomocą `out_inserted`, realokując w razie potrzeby, aby obsłużyć wzrost wykładniczy (choć MRV pomaga to ograniczyć).

