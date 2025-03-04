// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2025 all rights reserved


// externals
import React from 'react'
// locals
import styles from './styles'


// the area
const loading = () => (
    <section style={styles.loading}>
        <div style={styles.placeholder}>
            <svg style={styles.logo} version="1.1" xmlns="http://www.w3.org/2000/svg">
                <g fill="#f37f19" stroke="none">
                    <path d="M 100 0
                             C 200 75 -125 160 60 300
                             C 20 210 90 170 95 170
                             C 80 260 130 225 135 285
                             C 160 260 160 250 155 240
                             C 180 260 175 270 170 300
                             C 205 270 240 210 195 135
                             C 195 165 190 180 160 200
                             C 175 180 220 55 100 0
                             Z"/>
                </g>
            </svg>
            <p style={styles.message}>
                loading; please wait...
                <a href="/stop"> </a>
            </p>
        </div>
    </section>
)


// publish
export default loading


// end of file
