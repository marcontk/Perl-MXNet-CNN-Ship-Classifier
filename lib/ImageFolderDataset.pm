package ImageFolderDataset;

use strict;
use warnings;
use base 'AI::MXNet::Gluon::Data::Set';
use AI::MXNet qw(mx);
use AI::MXNet::Image;
use File::Find;
use File::Spec;
use List::Util qw(shuffle);

sub new {
    my ($class, $root, $transform) = @_;

    my (@data, %label_map);

    # Paso 1: recolectar todos los nombres de clase (subdirectorios)
    my %dirs;
    find(sub {
        return unless -f && /\.(jpg|jpeg|png)$/i;
        my @path_parts = File::Spec->splitdir($File::Find::dir);
        my $label_name = $path_parts[-1];
        $dirs{$label_name} = 1;
    }, $root);

    # Paso 2: asignar etiquetas numéricas ordenadas alfabéticamente
    my @class_names = sort keys %dirs;
    %label_map = map { $class_names[$_] => $_ } 0..$#class_names;

    # Paso 3: recolectar imágenes con sus etiquetas
    find(sub {
        return unless /\.(jpg|jpeg|png)$/i;
        my @dirs = File::Spec->splitdir($File::Find::dir);
        my $label_name = $dirs[-1];
        my $label = $label_map{$label_name};

        push @data, {
            path  => $File::Find::name,
            label => $label,
        };
    }, $root);

    @data = shuffle(@data);

    return bless {
        data      => \@data,
        transform => $transform,
        label_map => \%label_map,
        classes   => \@class_names,
    }, $class;
}

sub len {
    my $self = shift;
    return scalar @{ $self->{data} };
}

sub at {
    my ($self, $idx) = @_;
    my $entry = $self->{data}[$idx];
    my $img_path = $entry->{path};
    my $label    = $entry->{label};

    my $img = mx->img->imread($img_path);
    unless (defined $img) {
        warn "\x{26a0}\x{fe0f} No se pudo leer la imagen: $img_path\n";
        return (undef, undef, undef);
    }

    if ($self->{transform}) {
        ($img, $label) = $self->{transform}->($img, $label);
    }

    return ($img, $label, $img_path);
}

# Método adicional: obtener el nombre de clase desde el índice
sub class_name {
    my ($self, $label_id) = @_;
    return $self->{classes}[$label_id];
}

# Método adicional: número total de clases
sub num_classes {
    my $self = shift;
    return scalar @{ $self->{classes} };
}

__END__

=pod

=head1 NAME

ImageFolderDataset - A simple folder-based image dataset loader for AI::MXNet in Perl.

=head1 SYNOPSIS

  use ImageFolderDataset;

  my $dataset = ImageFolderDataset->new(
      "path/to/dataset",
      sub {
          my ($img, $label) = @_;
          # custom transform here
          return ($img, $label);
      }
  );

  my ($img, $label, $path) = $dataset->at(0);

=head1 DESCRIPTION

This module provides a lightweight dataset class modeled after the 
folder-based datasets commonly used in deep learning frameworks. 

It scans a root directory containing subdirectories, each representing
a class, and loads image paths along with their corresponding labels.

Images are read using C<AI::MXNet::Image> and can optionally be passed
through a user-defined transform.

=head1 FEATURES

=over 4

=item *
Automatic discovery of class names based on folder structure.

=item *
Alphabetical mapping of class names to numeric labels.

=item *
Randomized ordering of dataset samples.

=item *
Optional transform hook for preprocessing images.

=item *
Convenient C<class_name> and C<num_classes> helper methods.

=back

=head1 METHODS

=head2 new($root, $transform)

Creates the dataset by scanning C<$root>. Optionally accepts
a C<$transform> coderef that receives C<($img, $label)> and must
return modified versions.

=head2 len()

Returns the number of samples.

=head2 at($index)

Loads and returns C<(image, label, full_path)> for a given index.

=head1 AUTHOR

Marcelo Contreras C<< <marcontk@cpan.org> >>

=head1 LICENSE

This module is released under the same terms as Perl itself.

=cut

1;
