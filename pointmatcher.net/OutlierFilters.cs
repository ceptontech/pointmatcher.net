﻿using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

namespace pointmatcher.net
{
    public class TrimmedDistOutlierFilter : IOutlierFilter
    {
        float ratio;
        public TrimmedDistOutlierFilter(float ratio = 0.85f)
        {
            this.ratio = ratio;
        }

        public Matrix<float> ComputeOutlierWeights(DataPoints filteredReading, DataPoints filteredReference, Matches matches)
        {
            float limit = matches.GetDistsQuantile(this.ratio);
            Matrix<float> dists = new DenseMatrix(matches.Dists);
            var result = dists.Clone();
            result.MapInplace(d => (d <= limit) ? 1 : 0);
            return result;
        }
    }
}
