/**
 * ImportExample.java
 * 
 *
 * This file is part of Multi-Label-Topic-Modeling.
 * 
 * Multi-Label-Topic-Modeling is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 * 
 * Multi-Label-Topic-Modeling is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301 USA
 *
 */

package org.kramerlab;

import java.io.*;
import java.util.*;
import java.util.regex.*;

import cc.mallet.pipe.FeatureCountPipe;
import cc.mallet.pipe.CharSequence2CharNGrams;
import cc.mallet.pipe.CharSequence2TokenSequence;
import cc.mallet.pipe.FeatureSequence2FeatureVector;
import cc.mallet.pipe.Input2CharSequence;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.pipe.TargetStringToFeatures;
import cc.mallet.pipe.PrintInputAndTarget;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.Target2Label;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.pipe.TokenSequenceLowercase;
import cc.mallet.pipe.TokenSequenceRemoveStopwords;
import cc.mallet.pipe.iterator.FileIterator;
import cc.mallet.types.Alphabet;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;

public class ImportExample {

    Pipe pipe;
    Alphabet alphabet;

    public ImportExample() {
        pipe = buildPipe();
    }

    public Pipe buildPipe() {
        ArrayList pipeList = new ArrayList();


        pipeList.add(new TargetStringToFeatures());
                
        // Read data from File objects
        pipeList.add(new Input2CharSequence("UTF-8"));

        // Regular expression for what constitutes a token.
        // This pattern includes Unicode letters, Unicode numbers,
        // and the underscore character. Alternatives:
        // "\\S+" (anything not whitespace)
        // "\\w+" ( A-Z, a-z, 0-9, _ )
        // "[\\p{L}\\p{N}_]+|[\\p{P}]+" (a group of only letters and numbers OR
        // a group of only punctuation marks)
        Pattern tokenPattern = Pattern.compile("[\\p{L}\\p{N}_]+");

        // Tokenize raw strings
        pipeList.add(new CharSequence2TokenSequence(tokenPattern));

        // Normalize all tokens to all lowercase
        pipeList.add(new TokenSequenceLowercase());


        TokenSequenceRemoveStopwords tsrs = new TokenSequenceRemoveStopwords(false,false);
       
        // Remove stopwords from a standard English stoplist.
        // options: [case sensitive] [mark deletions]
        pipeList.add(tsrs);


        // Rather than storing tokens as strings, convert
        // them to integers by looking them up in an alphabet.
        pipeList.add(new TokenSequence2FeatureSequence());

        // Do the same thing for the "target" field:
        // convert a class label string to a Label object,
        // which has an index in a Label alphabet.
        // pipeList.add(new Target2Label());

        // Now convert the sequence of features to a sparse vector,
        // mapping feature IDs to counts.
        // pipeList.add(new FeatureSequence2FeatureVector());

        // Print out the features and the label
        // pipeList.add(new PrintInputAndTarget());
        SerialPipes pl = new SerialPipes(pipeList);

        return pl;
    }
	

    public InstanceList readDirectory(File directory) {
        return readDirectories(new File[] { directory });
    }

    public InstanceList readDirectories(File[] directories) {

        // Construct a file iterator, starting with the
        // specified directories, and recursing through subdirectories.
        // The second argument specifies a FileFilter to use to select
        // files within a directory.
        // The third argument is a Pattern that is applied to the
        // filename to produce a class label. In this case, I've
        // asked it to use the last directory name in the path.
        FileIterator iterator = new FileIterator(directories, new TxtFilter(),
                                                 FileIterator.LAST_DIRECTORY);

        // Construct a new instance list, passing it the pipe
        // we want to use to process// instances.
        InstanceList instances = new InstanceList(pipe);
        this.alphabet = pipe.getAlphabet();
        // alphabet.
        // Now process each instance provided by the iterator.
        instances.addThruPipe(iterator);

        return instances;
    }

    //sorts and InstanceList by the integer values of their names
    public InstanceList sortByID(InstanceList data){
        InstanceList result = new InstanceList();
        for(int i=0;i<data.size();++i){
            Instance inst = data.get(i);
            Integer name = Integer.valueOf((String)inst.getName());
            int j=0;
            while(j<result.size() && Integer.valueOf((String)result.get(j).getName())<name){
                j++;
            }
            result.add(j,inst);
        }

        return result;
    }


    public Instance readOneFile(String filename) {
        Instance inst = null;
        try {
            BufferedReader br = new BufferedReader(new FileReader(filename));
            String text;

            text = br.readLine();

            inst = new Instance(text, null, null, null);
            br.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return pipe.instanceFrom(inst);
    }

    public InstanceList readFromFile(String filename){
        
        try{
            CsvIterator csv=null;
       
            csv = new CsvIterator(new FileReader(filename),"([^\t]+)\t([^\t]+)\t(.*)",3,2,1);
            InstanceList instances= new InstanceList(pipe);
            instances.addThruPipe(csv);
            return sortByID(instances);
        }catch(FileNotFoundException e){
            e.printStackTrace();
        }
        return null;
    }

    public static void main(String[] args) throws IOException {

        ImportExample importer = new ImportExample();
        InstanceList instances = importer.readFromFile(args[0]);
        //		InstanceList instances = importer.readDirectory(new File(args[0]));
        instances.save(new File(args[1]));

    }

    /** This class illustrates how to build a simple file filter */
    class TxtFilter implements FileFilter {

        /**
         * Test whether the string representation of the file ends with the
         * correct extension. Note that {@ref FileIterator} will only call this
         * filter if the file is not a directory, so we do not need to test that
         * it is a file.
         */
        public boolean accept(File file) {
            return file.toString().endsWith(".txt");
        }
    }

}
