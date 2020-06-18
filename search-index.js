var searchIndex = JSON.parse('{\
"chfft":{"doc":"Chalharu\'s Fastest Fourier Transform.","i":[[3,"CFft1D","chfft","Perform a complex-to-complex one-dimensional Fourier…",null,null],[3,"CFft2D","","Perform a complex-to-complex two-dimensional Fourier…",null,null],[3,"Dct1D","","Perform a discrete cosine transform",null,null],[3,"Mdct1D","","Perform a Modified discrete cosine transform",null,null],[3,"RFft1D","","Perform a real-to-complex one-dimensional Fourier transform",null,null],[4,"DctType","","DCT Type",null,null],[13,"Two","","DCT-II",0,null],[13,"Three","","DCT-III",0,null],[11,"new","","Returns a instances to execute FFT",1,[[]]],[11,"with_len","","Returns a instances to execute length initialized FFT",1,[[]]],[11,"setup","","Reinitialize length",1,[[]]],[11,"forward","","The 1 scaling factor forward transform",1,[[],[["vec",3],["complex",3]]]],[11,"forward0","","The 1 scaling factor forward transform",1,[[],[["vec",3],["complex",3]]]],[11,"forwardu","","The \\\\(\\\\frac 1 {\\\\sqrt n}\\\\) scaling factor forward transform",1,[[],[["vec",3],["complex",3]]]],[11,"forwardn","","The \\\\(\\\\frac 1 {n}\\\\) scaling factor forward transform",1,[[],[["vec",3],["complex",3]]]],[11,"backward","","The \\\\(\\\\frac 1 n\\\\) scaling factor backward transform",1,[[],[["vec",3],["complex",3]]]],[11,"backward0","","The 1 scaling factor backward transform",1,[[],[["vec",3],["complex",3]]]],[11,"backwardu","","The \\\\(\\\\frac 1 {\\\\sqrt n}\\\\) scaling factor backward transform",1,[[],[["vec",3],["complex",3]]]],[11,"backwardn","","The \\\\(\\\\frac 1 n\\\\) scaling factor backward transform",1,[[],[["vec",3],["complex",3]]]],[11,"forward0i","","The 1 scaling factor and in-place forward transform",1,[[]]],[11,"backward0i","","The 1 scaling factor and in-place backward transform",1,[[]]],[11,"forwardui","","The \\\\(\\\\frac 1 {\\\\sqrt n}\\\\) scaling factor and in-place…",1,[[]]],[11,"backwardui","","The \\\\(\\\\frac 1 {\\\\sqrt n}\\\\) scaling factor and in-place…",1,[[]]],[11,"new","","Returns a instances to execute FFT",2,[[]]],[11,"with_len","","Returns a instances to execute length initialized FFT",2,[[]]],[11,"setup","","Reinitialize length",2,[[]]],[11,"forward","","The 1 scaling factor forward transform",2,[[],[["vec",3],["vec",3]]]],[11,"forward0","","The 1 scaling factor forward transform",2,[[],[["vec",3],["vec",3]]]],[11,"forwardu","","The \\\\(\\\\frac 1 {\\\\sqrt n}\\\\) scaling factor forward transform",2,[[],[["vec",3],["vec",3]]]],[11,"forwardn","","The \\\\(\\\\frac 1 n\\\\) scaling factor forward transform",2,[[],[["vec",3],["vec",3]]]],[11,"backward","","The \\\\(\\\\frac 1 n\\\\) scaling factor backward transform",2,[[],[["vec",3],["vec",3]]]],[11,"backward0","","The 1 scaling factor backward transform",2,[[],[["vec",3],["vec",3]]]],[11,"backwardu","","The \\\\(\\\\frac 1 {\\\\sqrt n}\\\\) scaling factor backward transform",2,[[],[["vec",3],["vec",3]]]],[11,"new","","Returns a instances to execute DCT",3,[[["dcttype",4]]]],[11,"setup","","Reinitialize length",3,[[["dcttype",4]]]],[11,"forward","","The 1 scaling factor forward transform",3,[[],["vec",3]]],[11,"forward0","","The 1 scaling factor forward transform",3,[[],["vec",3]]],[11,"forwardu","","The unitary transform scaling factor forward transform",3,[[],["vec",3]]],[11,"forwardn","","The inverse scaling factor forward transform",3,[[],["vec",3]]],[11,"with_sine","","Returns a instances to execute DCT with sine window",4,[[]]],[11,"with_vorbis","","Returns a instances to execute DCT with vorbis window",4,[[]]],[11,"new","","Returns a instances to execute DCT",4,[[]]],[11,"setup","","Reinitialize length",4,[[]]],[11,"forward","","The 1 scaling factor forward transform",4,[[],["vec",3]]],[11,"backward","","The 1 scaling factor backward transform",4,[[],["vec",3]]],[11,"forwardu","","The \\\\(\\\\sqrt{\\\\frac 2 n}\\\\) scaling factor forward transform",4,[[],["vec",3]]],[11,"backwardu","","The \\\\(\\\\sqrt{\\\\frac 2 n}\\\\) scaling factor backward transform",4,[[],["vec",3]]],[11,"new","","Returns a instances to execute FFT",5,[[]]],[11,"setup","","Reinitialize length",5,[[]]],[11,"forward","","The 1 scaling factor forward transform",5,[[],[["vec",3],["complex",3]]]],[11,"forward0","","The 1 scaling factor forward transform",5,[[],[["vec",3],["complex",3]]]],[11,"forwardu","","The \\\\(\\\\frac 1 {\\\\sqrt n}\\\\) scaling factor forward transform",5,[[],[["vec",3],["complex",3]]]],[11,"forwardn","","The \\\\(\\\\frac 1 n\\\\) scaling factor forward transform",5,[[],[["vec",3],["complex",3]]]],[11,"backward","","The \\\\(\\\\frac 1 n\\\\) scaling factor backward transform",5,[[],["vec",3]]],[11,"backward0","","The 1 scaling factor backward transform",5,[[],["vec",3]]],[11,"backwardu","","The \\\\(\\\\frac 1 {\\\\sqrt n}\\\\) scaling factor backward transform",5,[[],["vec",3]]],[11,"backwardn","","The \\\\(\\\\frac 1 n\\\\) scaling factor backward transform",5,[[],["vec",3]]],[11,"from","","",1,[[]]],[11,"into","","",1,[[]]],[11,"borrow","","",1,[[]]],[11,"try_from","","",1,[[],["result",4]]],[11,"try_into","","",1,[[],["result",4]]],[11,"borrow_mut","","",1,[[]]],[11,"type_id","","",1,[[],["typeid",3]]],[11,"from","","",2,[[]]],[11,"into","","",2,[[]]],[11,"borrow","","",2,[[]]],[11,"try_from","","",2,[[],["result",4]]],[11,"try_into","","",2,[[],["result",4]]],[11,"borrow_mut","","",2,[[]]],[11,"type_id","","",2,[[],["typeid",3]]],[11,"from","","",3,[[]]],[11,"into","","",3,[[]]],[11,"borrow","","",3,[[]]],[11,"try_from","","",3,[[],["result",4]]],[11,"try_into","","",3,[[],["result",4]]],[11,"borrow_mut","","",3,[[]]],[11,"type_id","","",3,[[],["typeid",3]]],[11,"from","","",4,[[]]],[11,"into","","",4,[[]]],[11,"borrow","","",4,[[]]],[11,"try_from","","",4,[[],["result",4]]],[11,"try_into","","",4,[[],["result",4]]],[11,"borrow_mut","","",4,[[]]],[11,"type_id","","",4,[[],["typeid",3]]],[11,"from","","",5,[[]]],[11,"into","","",5,[[]]],[11,"borrow","","",5,[[]]],[11,"try_from","","",5,[[],["result",4]]],[11,"try_into","","",5,[[],["result",4]]],[11,"borrow_mut","","",5,[[]]],[11,"type_id","","",5,[[],["typeid",3]]],[11,"from","","",0,[[]]],[11,"into","","",0,[[]]],[11,"to_owned","","",0,[[]]],[11,"clone_into","","",0,[[]]],[11,"borrow","","",0,[[]]],[11,"try_from","","",0,[[],["result",4]]],[11,"try_into","","",0,[[],["result",4]]],[11,"borrow_mut","","",0,[[]]],[11,"type_id","","",0,[[],["typeid",3]]],[11,"clone","","",0,[[],["dcttype",4]]],[11,"default","","Returns a instances to execute FFT",1,[[]]],[11,"default","","Returns a instances to execute FFT",2,[[]]],[11,"eq","","",0,[[["dcttype",4]]]],[11,"fmt","","",1,[[["formatter",3]],["result",6]]],[11,"fmt","","",2,[[["formatter",3]],["result",6]]],[11,"fmt","","",3,[[["formatter",3]],["result",6]]],[11,"fmt","","",0,[[["formatter",3]],["result",6]]],[11,"fmt","","",4,[[["formatter",3]],["result",6]]],[11,"fmt","","",5,[[["formatter",3]],["result",6]]]],"p":[[4,"DctType"],[3,"CFft1D"],[3,"CFft2D"],[3,"Dct1D"],[3,"Mdct1D"],[3,"RFft1D"]]}\
}');
addSearchOptions(searchIndex);initSearch(searchIndex);