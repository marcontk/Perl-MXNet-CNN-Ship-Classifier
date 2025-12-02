package ConfusionMatrixPlot;

use strict;
use warnings;
use Exporter 'import';
use Chart::Plotly;
use Chart::Plotly::Plot;
use Chart::Plotly::Trace::Heatmap;
#use utf8;
#use Encode qw(encode);

our @EXPORT_OK = qw(plot_confusion_matrix);

sub plot_confusion_matrix {
    my ($matrix_ref, $x_labels_ref, $y_labels_ref, $title) = @_;
    $title ||= 'Matriz de Confusión';

    # Preparar datos
    my @z = @$matrix_ref;
    my @x = @$x_labels_ref;
    my @y = @$y_labels_ref;

    # Generar texto para anotaciones
    my @annotations;
    for my $i (0..$#y) {
        for my $j (0..$#x) {
            push @annotations, {
                x => $x[$j],
                y => $y[$i],
                text => sprintf("%d", $z[$i][$j]),
                showarrow => JSON::PP::false,
                font => { color => 'white', size => 14 },
            };
        }
    }

    # Crear heatmap
    my $heatmap = Chart::Plotly::Trace::Heatmap->new(
        z => \@z,
        x => \@x,
        y => \@y,
        colorscale => 'Viridis',
        showscale => JSON::PP::true,
        hoverinfo => 'skip',
    );

    # Layout
    my $layout = {
        title => $title,
        xaxis => { title => 'Prediccion' },
        yaxis => { title => 'Real', autorange => 'reversed' },
        annotations => \@annotations,
        width => 400,
        height => 400,
    };

    # Plot
    my $plot = Chart::Plotly::Plot->new(
        traces => [$heatmap],
        layout => $layout
    );

    return $plot;
}

1;
