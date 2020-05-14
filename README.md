CSE 571 - Artificial Intelligence

Team Project: Topic 1 - Bidirectional search

Team members:
Juan Ortiz, Edward Meza, Juan Luna, Arun Shankar

The Meet in the Middle (MM) algorithm was implemented and integrated into the Pacman domain for path finding problems.

NOTE : Must be run using Python 2.7
A path finding problem can be tested with the following command:

python pacman.py -l <maze> -p SearchAgent -a fn=<algorithm>

Where maze can be: smallMaze, mediumMaze, bigMaze

Different algorithms can be used:
bfs - Breadth First Search
dfs - Depth First Search
astar - A* Search
ucs - Uniform Cost Search
bds - Bidirectional Search

For example to run the path finding problem on the mediumMaze using bidirectional search do the following:

python pacman.py -l mediumMaze -p SearchAgent -a fn=bds
