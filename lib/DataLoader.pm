package DataLoader;

use strict;
use warnings;
use AI::MXNet qw(mx);
use List::Util qw(shuffle);

sub new {
    my ($class, %args) = @_;
    my $self = bless {}, $class;

    $self->{dataset}    = $args{dataset};
    $self->{batch_size} = $args{batch_size} || 32;
    $self->{shuffle}    = $args{shuffle}    // 1;

    $self->reset;
    return $self;
}

sub reset {
    my ($self) = @_;
    $self->{indices} = [ 0 .. $self->{dataset}->len - 1 ];
    $self->{indices} = [ shuffle @{$self->{indices}} ] if $self->{shuffle};
    $self->{position} = 0;
}

sub next {
    my ($self) = @_;
    my $start = $self->{position};
    return unless $start < @{$self->{indices}};

    my $end = $start + $self->{batch_size} - 1;
    $end = $#{$self->{indices}} if $end > $#{$self->{indices}};
    my @batch_idx = @{$self->{indices}}[$start .. $end];

    my (@data, @labels, @paths);
    foreach my $i (@batch_idx) {
        my ($img, $label, $path) = $self->{dataset}->at($i);
        push @data, $img;
        push @labels, $label;
        push @paths, $path; # Opcional: almacenar las rutas de las imágenes
    }

    $self->{position} = $end + 1;

    my $data_nd   = mx->nd->stack(@data);              # data_nd tiene forma [batch_size, 3, H, W] según lo definido en transformer
    my $label_nd  = mx->nd->array(\@labels);           # [batch_size]

    return ($data_nd, $label_nd, \@paths); # Retorna también las rutas de las imágenes
}

__END__

=pod

=head1 NAME

DataLoader - A mini-batch data iterator for AI::MXNet training pipelines in Perl.

=head1 SYNOPSIS

  use DataLoader;

  my $loader = DataLoader->new(
      dataset    => $dataset,
      batch_size => 32,
      shuffle    => 1,
  );

  while (my ($data, $labels, $paths) = $loader->next) {
      # training loop here
  }

=head1 DESCRIPTION

This module implements a lightweight DataLoader similar in spirit to 
PyTorch's DataLoader. It provides shuffled mini-batches of samples from 
a dataset object that implements C<len> and C<at>.

It is specifically designed to work with image datasets and 
C<AI::MXNet> NDArrays in Perl, enabling practical training loops 
for convolutional neural networks and other deep learning models.

=head1 FEATURES

=over 4

=item *
Automatic batching with configurable batch size.

=item *
Optional shuffling at each C<reset>.

=item *
Returns data, labels, and optionally paths per batch.

=item *
Converts Perl arrayrefs into MXNet NDArrays automatically.

=back

=head1 METHODS

=head2 new(%args)

Creates a new DataLoader. Arguments:

=over 4

=item * C<dataset> — an object providing C<len> and C<at>.

=item * C<batch_size> — number of items per batch (default: 32).

=item * C<shuffle> — whether to randomize batch order (default: 1).

=back

=head2 reset()

Resets the internal iterator, optionally shuffling the indices.

=head2 next()

Retrieves the next batch as:

  ($data_nd, $label_nd, \@paths)

Returns undef when no samples remain.

=head1 AUTHOR

Marcelo Contreras Kohl

=head1 LICENSE

This module is free software; you may redistribute it and/or modify it 
under the same terms as Perl itself.

=cut

1;