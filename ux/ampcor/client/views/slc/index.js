// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// externals
import React from 'react'
// locals
import styles from './styles'


// explore the input SLCs
const slc = (props) => {
    // the raster size; hardwired, until this information is retrievable
    const rasterWidth = 10344
    const rasterHeight = 36864
    // our tile size
    const tileWidth = 500
    const tileHeight = 500
    // this is the shape of the slc
    // N.B.: this clips the tiles that ride on the right and bottom margin of the data set;
    // they aren't whole tiles, so they either need a dynamically generated margin or more
    // complicated request logic
    const cols = Math.floor(rasterWidth/tileWidth)
    const rows = Math.floor(rasterHeight/tileHeight)

    // add size information to the {plot{ style sheet
    styles.plot.width = `${cols*tileWidth}px`
    styles.plot.height = `${rows*tileHeight}px`
    // and to the {tile} sheet
    styles.tile.width = `${tileWidth}px`
    styles.tile.height = `${tileHeight}px`

    // the zoom level
    const zoom = 0
    // build the tile uri stem
    const uri = `/slc/ref/`
    // and the tile size spec
    const tileSpec = `+${tileHeight}x${tileWidth}`

    // tile specification factory
    const tspec = (first, ...rest) =>
        rest.reduce(
            (accum, val) => `${accum}x${val}`,
            `${first}`
        )

    // the cartesian product calculator
    const cartesian = (first, ...rest) =>
        rest.reduce(
            (prefix, arg, foo) =>
                prefix.flatMap( prefixEntry =>
                    arg.map(idx => [prefixEntry, shape[foo+1]*idx].flat())
                ),
            // convert {first} into an array of consecutive integers
            first.map(idx => shape[0]*idx)
        )

    // make an array with all the column indices
    const colIndices = Array(cols).fill().map((_, idx) => idx)
    // and another with all the row indices
    const rowIndices = Array(rows).fill().map((_, idx) => idx)
    // turn the tile shape into an array too
    const shape = [tileHeight, tileWidth]

    // build the tile specifications
    const tiles = cartesian(rowIndices, colIndices).map(
        origin => `tile-${zoom}@` + tspec(...origin) + `+${tileHeight}x${tileWidth}`
    )

    // build the container, fill it, and return it
    return (
        <section style={styles.slc}>
            <div style={styles.viewport}>
                <div style={styles.plot}>
                    {tiles.map(tile => (
                        <img key={`${tile}`} loading="lazy" src={uri+tile} style={styles.tile}/>
                    ))}
                </div>
            </div>
        </section>
    )
}


// publish
export default slc


// end of file
