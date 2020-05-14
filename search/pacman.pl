#!/opt/local/bin/perl


print "Hello!\n\n";


my $layoutFile = 'layouts.txt';
open(my $lh, '<:encoding(UTF-8)', $layoutFile) or die "Could not open file '$layoutFile' $!";

my $outputFile = 'output.csv';
open(my $oh, '>', $outputFile) or die "Could not open file '$outputFile' $!\n";

print $oh "Layout,Algorithm,Nodes_Expanded,Path_Cost\n";

my @algos =    ('bfs', 
				'dfs', 
				'astar', 
				#'ucs', 
				'bds'
				);




while ($row = <$lh>) {
	chomp $row;
	$row =~ s/(\w+)\..../$1/g;
	chomp $row;
	for $algo (@algos) {
		$output = `python pacman.py -q -l $row -p SearchAgent -a fn=$algo`;
		print "python pacman.py -q -l $row -p SearchAgent -a fn=$algo\n";
		$out1 = $output;
		$out2 = $output;
		$out1 =~ s/Search nodes expanded: (\d+)/$1/g;
		$nodes = $1;
		print "Nodes expanded for layout $row using search algorithm $algo\: $nodes\n";
		$out2 =~ s/Path found with total cost of (\d+)/$1/g;
		$pathcost = $1;
		print "Total cost of path found for layout $row using search algorithm $algo\: $pathcost\n\n";
		print $oh "$row,$algo,$nodes,$pathcost\n"

		
	}
	
}

close $oh;

print "done\n";
