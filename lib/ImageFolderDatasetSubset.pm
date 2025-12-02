package ImageFolderDatasetSubset;
use strict;
use warnings;
use Scalar::Util qw(blessed);
use base 'AI::MXNet::Gluon::Data::Set';

sub new {
    my ($class, $data, $labels) = @_;
    bless { data => $data, labels => $labels }, $class;
}

sub len {
    my $self = shift;
    scalar(@{$self->{data}});
}

sub at {
    my ($self, $idx) = @_;
    #return ($self->{data}[$idx], $self->{labels}[$idx]);
    my $entry = $self->{data}[$idx];
    return ($entry->{img}, $entry->{label}, $entry->{path});
}

sub as_ndarray_pairs {
    my ($self, $ctx) = @_;
    $ctx //= AI::MXNet::Context->current_ctx;

    my @pares;
    my $images = $self->{data};
    my $labels = $self->{labels};

    for my $i (0..$#$images) {
        my $img = $images->[$i];
        my $lbl = $labels->[$i];

        my $nd_img = (blessed($img) && $img->isa("AI::MXNet::NDArray")) 
                     ? $img 
                     : AI::MXNet::NDArray->array($img, ctx => $ctx);

        #my $nd_lbl = AI::MXNet::NDArray->array([$lbl], ctx => $ctx);
        my $nd_lbl = AI::MXNet::NDArray->array([$lbl], ctx => $ctx)->asscalar;

        push @pares, [$nd_img, $nd_lbl];
    }

    return \@pares;
}


__END__

=pod

=head1 NAME

ImageFolderDatasetSubset - Subset view for image datasets used with AI::MXNet in Perl.

=head1 SYNOPSIS

  use ImageFolderDatasetSubset;

  my $subset = ImageFolderDatasetSubset->new($data_ref, $labels_ref);

  my ($img, $label, $path) = $subset->at(0);

=head1 DESCRIPTION

This module provides a lightweight subset wrapper for datasets used in 
image classification workflows with C<AI::MXNet> in Perl. It allows you to:

=over 4

=item *
Split datasets into train/validation/test subsets without duplicating images.

=item *
Preserve the internal order and metadata of each sample.

=item *
Expose a Gluon-compatible API, including C<len>, C<at>, and C<as_ndarray_pairs>.

=back

It is intended to work together with C<ImageFolderDataset.pm> and custom 
C<DataLoader> modules commonly used in Perl + MXNet training pipelines.

=head1 METHODS

=head2 new($data_arrayref, $labels_arrayref)

Creates a subset object from arrayrefs of data entries and labels.

=head2 len()

Returns the number of items in the subset.

=head2 at($index)

Returns C<(image, label, path)> for the given index. The C<image> can be
already an NDArray or a raw Perl structure depending on earlier processing.

=head2 as_ndarray_pairs($ctx)

Converts all pairs into NDArrays in the given MXNet context. Useful for 
batching or manual iteration outside a DataLoader.

=head1 AUTHOR

Marcelo Contreras C<< <marcontk@cpan.org> >>

=head1 LICENSE

This library is free software; you may redistribute it and/or modify it 
under the same terms as Perl itself.

=cut
1;
