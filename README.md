# VT ECE 5504 Final Project S2026

This repository contains all fines necessary to replicate the experiment described in the report. 

## To replicate
1. Install gcc or g++ depending on which benchmark version you intend to compile. .C versions are recommended.

2. Compile using the commands:
- gcc -O3 -fopenmp -static -march=x86-64 -mtune=generic GEMM.c -o GEMM
- gcc -O3 -fopenmp -static -march=x86-64 -mtune=generic FASP.c -o FASP

3. Move all files inside simulation\ and compiled binaries to simulation directory.

4. Run command:
- python allCacheConfigs.py

This will run the GEMM binary first by default. Proceed once this is complete.

5. Rename the results folder that was created called 'finalProj' in the simulation directory. This prevents it from being replaced.

6. Change 'default_binary' in simulate.py to FASP.

7. Run command:
- python allCacheConfigs.py

8. Rename the results folder that was created called 'finalProj' in the simulation directory for convenience.

9. You should now have two sets of 27 results like in Results\

## Example Results
The results used for analysis in the report are included under the Results\ folder.

## Precompiled Binaries
Precompiled binaries for x86 Windows are included under Precompiled (win)\


## Disclaimer
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## License
Not for use outside of the ECE 5504 course during the Spring semester of 2026.