package org.kramerlab.util;


import org.apache.commons.cli.*;
import java.util.ListIterator;

public class ExtendedGnuParser extends GnuParser {

    private boolean ignoreUnrecognizedOption;

    public ExtendedGnuParser(final boolean ignoreUnrecognizedOption) {
        this.ignoreUnrecognizedOption = ignoreUnrecognizedOption;
    }

    @Override
    protected void processOption(final String arg, final ListIterator iter) throws     ParseException {
        boolean hasOption = getOptions().hasOption(arg);
        if (hasOption || !ignoreUnrecognizedOption) {
            super.processOption(arg, iter);
        }
    }

}
