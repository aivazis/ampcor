// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2022 all rights reserved


// get colors
import { wheel, theme } from '~/palette'


// tool styling; see '~/widgets/tool/*.js' for details
const tool = {
    // the base state
    base: {
        box: {
            flex: "none",
            width: "2em",
            height: "2em",
            color: theme.banner.name,
            // backgroundColor: theme.page.background,
        },

        group: {
            transform: "scale(0.4)",
        },

        frame: {
            fill: wheel.gray.obsidian,
            fillOpacity: 0.4,
            stroke: wheel.gray.steel,
            strokeOpacity: 0.75,
            strokeWidth: "5px",
        },

        icon: {
            fontFamily: "inconsolata",
            fontSize: "50px",
            fill: theme.banner.name,
            stroke: theme.banner.name,
            strokeOpacity: 0.75,
            strokeWidth: "1px",
        },

    },
    inactive: {
        frame: {
            fillOpacity: .2,
        },
        icon: {
            strokeOpacity: .2,
        },
    },
    selected: {
        frame: {
            fillOpacity: 1.0,
            stroke: theme.banner.name,
        },
    }
}


// publish
export default {
    toolbox: {
        // for my children
        display: "flex",
        // vertical box, by default
        flexDirection: "row",
        // spread them out
        justifyContent: "space-between",
    },

    // tool styling; see '~/widgets/tool/*.js' for details
    tool: {
        // start with the default established up top
        ...tool
    },
}


// end of file
